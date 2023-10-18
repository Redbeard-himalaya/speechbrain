#!/bin/bash

ROOT_DIR=$(dirname $(dirname $(realpath ${0})))
SCRIPT_NAME=$(basename ${0})

function show_usage {
    echo "Usage: ${SCRIPT_NAME} [-h] show this help info."
    echo "       ${SCRIPT_NAME} [options] video [model options]"
    echo "${SCRIPT_NAME} remove background noises from video and save denoised video or audio"
    echo
    echo "  input:"
    echo "  video     Local file path to video."
    echo
    echo "  output:"
    echo "  video     Output video is saved in the same folder as input video with a"
    echo "            name equaling to video name's stem + _denoised + video suffix."
    echo "  audio     Output audio is saved in the same folder as input video with a"
    echo "            name equaling to video name's stem + _denoised + .wav."
    echo
    echo "  options:"
    echo "  -a        Output is saved as audio."
    echo "  -i        Enable network in docker container."
    echo "  -h        Show this help info."
    echo
    echo "  model options:"
    echo "  -m, --model_id <id>     The model id. Default 0."
    echo "  -v, --verbose               Show detailed log."
    echo
}


# ### main ###
# show usage by default
if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

NET_ENABLED=false
OUT_AUDIO=false
OPTIONS=''

# Reset in case getopts has been used previously in the shell.
OPTIND=1
#while getopts "hvc:s:t:w:" opt; do
while getopts "ahi" opt; do
    case ${opt} in
        a ) # process option a
            OUT_AUDIO=true
            ;;
        h ) # process option h
            show_usage
            exit 0
            ;;
        i ) # process option i
            NET_ENABLED=true
            ;;
        \? ) # invalid option
            echo
            show_usage
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

INPUT=$(realpath ${1})
shift
OPTIONS="${OPTIONS} ${@}"

INPUT_NAME=$(basename ${INPUT})
INPUT_NAME_STEM="${INPUT_NAME%.*}"
INPUT_NAME_SUFF="${INPUT_NAME##*.}"
INPUT_DIR=$(dirname ${INPUT})
if $OUT_AUDIO ; then
    OUTPUT_NAME=${INPUT_NAME_STEM}_denoised.wav
else
    OUTPUT_NAME=${INPUT_NAME_STEM}_denoised.${INPUT_NAME_SUFF}
fi
OUTPUT=${INPUT_DIR}/${OUTPUT_NAME}
OUTPUT_AUDIO_NAME=${VIDEO_NAME_STEM}_denoised.wav
OUTPUT_AUDIO=${VIDEO_DIR}/${OUTPUT_VIDEO_NAME}
DOCER_SERVICE=enhancement
CWD=$(pwd)


# create an empty output to be mounted by docker
if ! touch ${OUTPUT}; then
    exit 1
fi

if ${NET_ENABLED}; then
    DOCER_SERVICE=${DOCER_SERVICE}-with-net
fi

cd ${ROOT_DIR}
docker-compose -f docker/docker-compose.yml run --rm -v ${INPUT}:/video/${INPUT_NAME}:ro -v ${OUTPUT}:/video/${OUTPUT_NAME} ${DOCER_SERVICE} python /app/speech_enhance.py ${OPTIONS} /video/${INPUT_NAME} /video/${OUTPUT_NAME}
cd ${CWD}
