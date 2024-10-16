import json

data = {  "test_file":{
             "file_path": "test",
             "court_name": None,
    
             "case_id": 0,

             "case_year": 0,
    
             "judges_names": [],
    
             "parties_involved": {
                 "appellants": [],
                 "respondents": []
             },
    
              "sections": [],
    
              "acts": [],
    
             "summary":None,
    
              "compressed_text": None,
    
             "similar_cases": []
              }
              
          }

# Specify the file path where you want to save the JSON file
file_path = 'testing.json'

# Write the data to the JSON file
with open(file_path, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Data saved to '{file_path}'")
