# serializer.py
import json

class JsonSerializer:
    @staticmethod
    def serialize_to_json(data, file_name):
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4, default=str)
