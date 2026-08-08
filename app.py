from predictions.predict import (
    initialize_prediction_service,
    predict_all
)
initialize_prediction_service()
predictions = predict_all()
print(predictions)