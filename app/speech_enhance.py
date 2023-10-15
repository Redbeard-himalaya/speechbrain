import logging
from mimetypes import guess_type
from pathlib import Path
from shutil import copy
import subprocess
import tempfile
from typing import List, Union, Set

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

class SpeechEnhancer:
    MODELS = {
        0: "speechbrain/sepformer-wham16k-enhancement",
        1: "speechbrain/sepformer-dns4-16k-enhancement",
    }
    FFMPEG_SEG = "ffpb" \
        " -i {in_wav}" \
        " -f segment -segment_time 40" \
        " -c copy" \
        " {out_dir}/seg%03d.wav"
    FFMPEG_CONCAT = "ffpb" \
        " -f concat" \
        " -safe 0" \
        " -i {concat_file}" \
        " -c copy" \
        " {out_wav}"
    FFMPEG_EXTRACT_WAV = "ffpb" \
        " -i {in_video}" \
        " {out_wav}"
    FFMPEG_REPLACE_AUDIO = "ffpb -y" \
        " -i {in_video}" \
        " -i {in_wav}" \
        " -c:v copy" \
        " -map 0:v:0" \
        " -map 1:a:0" \
        " {out_video}"


    def __init__(self,
                 model_id: int = 0,
                 model_dir: Path = Path("/models"),
                 logger = logging,
                 download_only = False,
    ):
        model_name = Path(self.MODELS[model_id]).name
        self.model = separator.from_hparams(source=self.MODELS[model_id],
                                            savedir=str(model_dir.resolve()/model_name),
                                            run_opts={"device":"cuda"},
                                            download_only=download_only,
        )
        self.logging = logger


    def Separate(self, in_media: Path, out_media: Path):
        in_media_is_video = guess_type(in_media)[0].startswith("video")
        out_media_is_video = guess_type(out_media)[0].startswith("video")
        if not in_media_is_video and out_media_is_video:
            self.logging.error("cat not generate video from audio")
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
        # with out_media.parent / "seg" as tmp_dir:
            tmp_dir = Path(tmp_dir)
            in_wav = in_media
            if in_media_is_video:
                in_wav = self._extract_wav(in_video=in_media, tmp_dir=tmp_dir)
            segs = self._seg(in_wav, tmp_dir)
            denoised_segs = [seg.parent / f"{seg.stem}_denoised{seg.suffix}" for seg in segs]
            for seg, denoised_seg in zip(segs, denoised_segs):
                enhanced = self.model.separate_file(path=str(seg.resolve()))
                torchaudio.save(str(denoised_seg.resolve()),
                                enhanced[:, :, 0].detach().cpu(),
                                16000)
            concat_file = self._concat_file(denoised_segs, tmp_dir)
            merged_wav = self._merge(concat_file, tmp_dir)
            if out_media_is_video:
                denoised_video = self._replace_audio(in_media, merged_wav, out_media)
            else:
                copy(merged_wav, out_media)


    def _seg(self, in_wav: Path, tmp_dir: Path) -> Set[Path]:
        cmd = self.FFMPEG_SEG.format(in_wav=in_wav, out_dir=tmp_dir)
        try:
            self.logging.debug(f"audio seg cmd: {cmd}")
            p = subprocess.run(cmd, shell=True, capture_output=False, check=True)
            return set(tmp_dir.glob('seg*.wav'))
        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode("utf-8").strip()
            self.logging.error(f"ffmpeg returns {e.returncode}, console [{msg}]")
            raise e
        except Exception as e:
            self.logging.error(f"exception in ffmpeg. e={repr(e)}")
            raise e


    def _concat_file(self, seg_files: Set[Path], tmp_dir: Path) -> Path:
        concat_file = tmp_dir / "concat.txt"
        with open(concat_file, "+w") as f:
            f.writelines(["file " + str(fp) + "\n" for fp in sorted(seg_files)])
        return concat_file


    def _merge(self, concat_file: Path, tmp_dir: Path) -> Path:
        out_wav = tmp_dir / "merged.wav"
        cmd = self.FFMPEG_CONCAT.format(concat_file=concat_file, out_wav=out_wav)
        try:
            self.logging.debug(f"audio merge cmd: {cmd}")
            p = subprocess.run(cmd, shell=True, capture_output=False, check=True)
            return out_wav
        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode("utf-8").strip()
            self.logging.error(f"ffmpeg returns {e.returncode}, console [{msg}]")
            raise e
        except Exception as e:
            self.logging.error(f"exception in ffmpeg. e={repr(e)}")
            raise e


    def _extract_wav(self, in_video: Path, tmp_dir: Path) -> Path:
        out_wav = tmp_dir / "ori.wav"
        cmd = self.FFMPEG_EXTRACT_WAV.format(in_video=in_video, out_wav=out_wav)
        try:
            self.logging.debug(f"audio extract cmd: {cmd}")
            p = subprocess.run(cmd, shell=True, capture_output=False, check=True)
            return out_wav
        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode("utf-8").strip()
            self.logging.error(f"ffmpeg returns {e.returncode}, console [{msg}]")
            raise e
        except Exception as e:
            self.logging.error(f"exception in ffmpeg. e={repr(e)}")
            raise e


    def _replace_audio(self, in_video: Path, in_wav: Path, out_video: Path):
        cmd = self.FFMPEG_REPLACE_AUDIO.format(in_video=in_video,
                                               in_wav=in_wav,
                                               out_video=out_video)
        try:
            self.logging.debug(f"audio replace cmd: {cmd}")
            p = subprocess.run(cmd, shell=True, capture_output=False, check=True)
            return out_video
        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode("utf-8").strip()
            self.logging.error(f"ffmpeg returns {e.returncode}, console [{msg}]")
            raise e
        except Exception as e:
            self.logging.error(f"exception in ffmpeg. e={repr(e)}")
            raise e


if __name__ == "__main__":
    '''
    Run this py file in container by:
    cd /tpp
    python3.10 src/speech_enhance.py in_wav out_wav
    '''

    import argparse
    import sys

    parser = argparse.ArgumentParser(description="run speechbrain speech enhancement.")
    parser.add_argument("in_wav", metavar="in_wav",
                        type=Path, help="the input wav")
    parser.add_argument("out_wav", metavar="out_wav",
                        type=Path, help="the output wav")
    parser.add_argument("-m", "--model_id", nargs="?",
                        choices=SpeechEnhancer.MODELS.keys(),
                        type=int, default=0,
                        help=f"available models {SpeechEnhancer.MODELS}")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    enhancer = SpeechEnhancer(model_id=args.model_id)
    try:
        enhancer.Separate(args.in_wav, args.out_wav)
    except Exception as e:
        logging.error(e)
        sys.exit(1)
    sys.exit()
