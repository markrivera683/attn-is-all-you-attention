import torch

from attention_regression.data import iter_episodes


def test_episodic_generator_shapes_and_seed_stability():
    episodes = iter_episodes(
        num_episodes=2,
        dim=5,
        memory_size=7,
        query_size=3,
        noise=0.1,
        seed=123,
    )
    repeated = iter_episodes(
        num_episodes=2,
        dim=5,
        memory_size=7,
        query_size=3,
        noise=0.1,
        seed=123,
    )

    first = episodes[0]
    assert first.X_m.shape == (7, 5)
    assert first.y_m.shape == (7,)
    assert first.X_q.shape == (3, 5)
    assert first.y_q.shape == (3,)
    assert first.w.shape == (5,)

    assert not torch.allclose(episodes[0].w, episodes[1].w)
    assert torch.allclose(episodes[0].X_m, repeated[0].X_m)
    assert torch.allclose(episodes[0].y_m, repeated[0].y_m)
    assert torch.allclose(episodes[0].X_q, repeated[0].X_q)
    assert torch.allclose(episodes[0].y_q, repeated[0].y_q)
    assert torch.allclose(episodes[0].w, repeated[0].w)
