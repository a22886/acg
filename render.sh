#!/bin/bash
set -e
frames=60
outdir=output5/
outvid=output.mp4
rm -rf "$outdir" "$outvid"
mkdir -p "$outdir" "$outdir"ply "$outdir"png "$outdir"obj
python main.py "$frames" "$outdir"
echo "Started reconstructing at " $(date +%X)
pysplashsurf reconstruct "$outdir"ply/fluid{}.ply -o "$outdir"obj/fluid{}.obj -s=0 -e=$((frames-1)) -n=$(nproc) -r=0.01 -l=3.5 -c=0.5 -q
echo "Finished reconstructing at " $(date +%X)
# python render.py "$outdir"fluid "$outdir" 0 $frames
for i in {0..4}; do
    python render.py $((i*frames/5)) $(((i+1)*frames/5)) "$outdir"png/ "$outdir"obj/fluid "$outdir"obj/rigid0_ "$outdir"obj/rigid1_ &
done
wait
# rm -rf output.mp4
ffmpeg -loglevel quiet -framerate 30 -i "$outdir"png/%d.png -c:v libx264 -pix_fmt yuv420p -vframes $frames "$outvid"

