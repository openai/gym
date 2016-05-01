ENVS = ["Ant-v0", "HalfCheetah-v0", "Hopper-v0", "Humanoid-v0", "InvertedDoublePendulum-v0", "Reacher-v0", "Swimmer-v0", "Walker2d-v0"]
OLD_COMMIT = "HEAD"

# ================================================================

import subprocess, gym
from gym import utils
from os import path

def cap(cmd):
    "Call and print command"
    print utils.colorize(cmd, "green")
    subprocess.check_call(cmd,shell=True)

# ================================================================

gymroot = path.abspath(path.dirname(path.dirname(gym.__file__)))
oldgymroot = "/tmp/old-gym"
comparedir = "/tmp/gym-comparison"

oldgymbase = path.basename(oldgymroot)

print "gym root", gymroot
thisdir = path.abspath(path.dirname(__file__))
print "this directory", thisdir
cap("rm -rf %(oldgymroot)s %(comparedir)s && mkdir %(comparedir)s && cd /tmp && git clone %(gymroot)s %(oldgymbase)s"%locals())
for env in ENVS:
    print utils.colorize("*"*50 + "\nENV: %s" % env, "red")
    writescript = path.join(thisdir, "write_rollout_data.py")
    outfileA = path.join(comparedir, env) + "-A.npz"
    cap("python %(writescript)s %(env)s %(outfileA)s"%locals())
    outfileB = path.join(comparedir, env) + "-B.npz"
    cap("python %(writescript)s %(env)s %(outfileB)s --gymdir=%(oldgymroot)s"%locals())

    comparescript = path.join(thisdir, "compare_rollout_data.py")
    cap("python %(comparescript)s %(outfileA)s %(outfileB)s"%locals())

