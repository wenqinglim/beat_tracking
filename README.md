# Beat Tracking

Implementation of Ellis, 2007 [1] beat tracking, including
- Onset Strength Envelope (as described in paper)
- Tempo Estimation (as described in paper)
- Dynamic Programming beat tracking (as described in paper)
- Downbeat detection (based on onset strength envelope and beats identified)

[1] Ellis, D. P. (2007). Beat tracking by dynamic programming. Journal of New Music Research, 36(1), 51-60.

## Environment setup
In a Python virtual environment:
```
pip install -r requirements.txt
```

## Running the beat tracker
1. Ensure data is in data folders
- `BallroomData/` - Ballroom data from http://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz should be placed here
- `BallroomAnnotations-master/` - Ballroom Annotations data from https://github.com/CPJKU/BallroomAnnotations should be placed here (not required for beat tracker, but required in `beat_tracking_eval.ipynb` for evaluation)

2. Run beat tracker:
```
import beatTracker from  beat_tracker

filename = "BallroomData/Quickstep/Albums-Ballroom_Classics4-20.wav"
beats, downbeats = beatTracker(filename)
```


## Directory Description

- `beat_tracker.py` - main script for beat tracking implementation
- `beat_tracking_ellis.ipynb` - notebook for beat tracking implementation
- `beat_tracking_eval.ipynb` - notebook for evaluating beat tracking on ballroom annotation



