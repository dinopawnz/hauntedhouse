import sys, os
sys.path.insert(0, "/home/claude/hh13/models"); sys.path.insert(0, "/home/claude/hh13/tools")
import props
res = float(os.environ.get("BLOOD_RES", "0.034"))
props.blood("blood_floor", wall=False, seed=3, res=res); props.blood("blood_floor_b", wall=False, seed=8, res=res); props.blood("blood_wall", wall=True, seed=5, res=res)
