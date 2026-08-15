from src.data.features.feature_store.hopswork_client import connect


FEATURE_GROUP_NAME = "aqi_training_features"
FEATURE_GROUP_VERSION = 2

FEATURE_VIEW_NAME = "aqi_prediction_fv"
FEATURE_VIEW_VERSION = 1


def get_feature_view():

    project = connect()

    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    query = fg.select_all()

    feature_view = fs.get_or_create_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
        query=query,
        description="Feature view for AQI prediction"
    )

    return feature_view