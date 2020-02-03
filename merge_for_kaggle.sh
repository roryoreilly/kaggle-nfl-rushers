 #!/bin/sh

num=$(ls submissions | wc -l | sed -e 's/^[[:space:]]*//')
cat ./rusher_yards/feature_extractor.py \
     ./rusher_yards/model.py \
     ./rusher_yards/train.py \
     ./rusher_yards/app.py \
     > ./submissions/submissions-"${num}".py
ls submissions | wc -l