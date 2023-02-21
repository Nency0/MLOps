from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer


def test_temporal_variable_transformer(sample_input_data):
    # Given
    extracter = ExtractLetterTransformer(
        variables=config.model_config.cabin_var_imputation
    )
    assert sample_input_data["cabin"].iat[246] == 'C106'

    # When
    subject = extracter.fit_transform(sample_input_data)

    # Then
    assert subject["cabin"].iat[246] == "C"
