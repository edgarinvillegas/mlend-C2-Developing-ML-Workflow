######################## SERIALIZE LAMBDA ####################
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3
    :param event: {
      "s3_bucket": "project2-scones",
      "s3_key": "test/bicycle_s_000513.png"
    }
    :returns image_data: Image serialized as base 64
    
    """

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    local_temp_path = '/tmp/image.png'
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, key, local_temp_path)

    # We read the data from a file
    with open(local_temp_path, "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    # print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data
        }
    }




######################## CLASSIFY LAMBDA ####################
import os
import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer
from sagemaker.predictor import Predictor


def lambda_handler(event, context):
    '''
    Classifies an image
    :param event: {
        "image_data": "iVBORw0KGgoAAAANSUhEUgAAACAAAA...",
    }
    :returns: The probability of each class as array
    '''
    # Fill this in with the name of your deployed model
    endpoint = event.get('endpoint') or os.getenv('ENDPOINT')

    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Instantiate a Predictor
    predictor = Predictor(
        endpoint_name=endpoint,
        sagemaker_session=sagemaker.session.Session(),
        # For this model the IdentitySerializer needs to be "image/png"
        serializer=IdentitySerializer("image/png")
    )

    # Make a prediction:
    str_inferences = predictor.predict(image).decode('utf-8')

    # We return the data back to the Step Function
    # event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': {
            'inferences': json.loads(str_inferences)
        }
    }



######################## EVALUATE THRESHOLD LAMBDA ####################
import json

THRESHOLD = .8

def lambda_handler(event, context):
    '''
    Raises an error if the inference is below the threshold (0.8).
    Otherwise returns empty object
    :param event: {
        "inferences": [0.6, 0.4],
    }
    :returns: {} or raises error
    '''    
    # Grab the inferences from the event
    inferences = event['inferences']

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any([x > THRESHOLD for x in inferences])

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': {}
    }
