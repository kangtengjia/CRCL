from model.SGRAF import EncoderSimilarity


def test_encoder_similarity_uses_configured_point_patch_count():
    encoder = EncoderSimilarity(embed_size=8, sim_dim=4, module_name="SGR", sgr_step=1, num_regions=200)

    assert encoder.v_global_w.embedding_local[1].num_features == 200
