#!/bin/bash
set -e
trap 'if [[ $? -ne 0 ]]; then kill 0; fi' EXIT

frames=$(sed -n 's/^frames *= *\([0-9]*\).*$/\1/p' main.py)
# frames=46
has_fluid=$(sed -n 's/^has_fluid *= *\(True\|False\).*$/\1/p' main.py)
fluid_obj_prefix=$(sed -n 's/^fluid_obj_prefix *= *"\(.*\)".*$/\1/p' main.py)
fluid_ply_prefix=$(sed -n 's/^fluid_ply_prefix *= *"\(.*\)".*$/\1/p' main.py)
output_png_prefix=$(sed -n 's/^output_png_prefix *= *"\(.*\)".*$/\1/p' main.py)
output_video=$(sed -n 's/^output_video *= *"\(.*\)".*$/\1/p' main.py)

python main.py

if [ "$has_fluid" == "True" ]; then
    echo "Started reconstructing at " $(date +%X)
    pysplashsurf reconstruct "${fluid_ply_prefix}{}.ply" -o "$fluid_obj_prefix"{}.obj -s=0 -e=$((frames-1)) -n=$(nproc) -r=0.01 -l=3.5 -c=0.5 -q
    echo "Finished reconstructing at " $(date +%X)
fi

for i in {0..4}; do
    python render.py $((i*frames/5)) $(((i+1)*frames/5)) &
done
wait

ffmpeg -loglevel quiet -framerate 30 -i "$output_png_prefix"%d.png -c:v libx264 -pix_fmt yuv420p -vframes "$frames" "$output_video"

