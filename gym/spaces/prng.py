import numpy

np_random = numpy.random.RandomState()

def seed(seed=None):
    """Seed the common numpy.random.RandomState used in spaces

    CF
    https://github.com/openai/gym/commit/58e6aa95e5af2c738557431f812abb81c505a7cf#commitcomment-17669277
    for some details about why we seed the spaces separately from the
    envs, but tl;dr is that it's pretty uncommon for them to be used
    within an actual algorithm, and the code becomes simpler to just
    use this common numpy.random.RandomState.
    """
    np_random.seed(seed)

# This numpy.random.RandomState gets used in all spaces for their
# 'sample' method. It's not really expected that people will be using
# these in their algorithms.
seed(0)
