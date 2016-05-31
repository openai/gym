from gym.scoreboard import registration

def test_correct_registration():
    try:
        registration.registry.finalize(strict=True)
    except registration.RegistrationError as e:
        assert False, "Caught: {}".format(e)
