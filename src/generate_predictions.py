# from src.model_deployment import get_selected_model_info
# import os
# from google.cloud import bigquery, aiplatform
# from typing import Optional
# import datetime
# import argparse

# class GeneratePredictions:
#     def __init__(self):
#         """
#         Initialize BigQuery ML model training.
#         """
#         if os.environ.get("AIRFLOW_HOME"):
#             # Running locally or through Airflow
#             os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"

#         self.client = bigquery.Client()
#         self.project_id = self.client.project
#         self.dataset_id = "books"
#         self.location = "us-central1"
#         aiplatform.init(
#             project=self.project_id,
#             location=self.location
#         )

#     def get_mf_predictions(self, model_name, user_id):
#         """
#         Generate book recommendations for a given user_id using Matrix Factorization model.
#         """
#         config = {
#             "project_id": self.project_id,
#             "dataset": self.dataset_id,
#             "model_name": model_name
#         }
        
#         with open("src/mf_predictor_query.sql", "r") as file:
#             query_template = file.read()
        
#         query = query_template.format(**config)
#         job_config = bigquery.QueryJobConfig(
#             query_parameters=[
#                 bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
#             ]
#         )
#         results = self.client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=False)
#         return results[['book_id', 'title', 'rating', 'author_names']]

#     def get_bt_predictions(self, model_name, user_id):
#         """
#         Generate book recommendations for a given user_id using Boosted Tree model.
#         """
#         config = {
#             "project_id": self.project_id,
#             "dataset": self.dataset_id,
#             "model_name": model_name
#         }
        
#         with open("src/bt_predictor_query.sql", "r") as file:
#             query_template = file.read()
        
#         query = query_template.format(**config)
#         job_config = bigquery.QueryJobConfig(
#             query_parameters=[
#                 bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
#             ]
#         )
#         results = self.client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=False)
#         print(results.columns)
#         return results[['book_id', 'title', 'rating', 'author_names']]
    
#     def get_version(self, display_name):
#         try:
#             models = aiplatform.Model.list(
#                 filter=f'display_name="{display_name}"',
#                 location=self.location
#             )
            
#             if not models:
#                 print(f"No model found with display name: {display_name}")
#                 return None
            
#             parent_model = models[0]
#             print(f"Found parent model: {parent_model.resource_name}")
            
#             versions = parent_model.versioning_registry.list_versions()
#             default_version = None
                
#             for v in versions:
#                 if hasattr(v, 'version_aliases') and 'default' in v.version_aliases:
#                     default_version = v
#                     break
#             return default_version.version_id if default_version else None
#         except Exception as e:
#             print(f"Error retrieving model version: {e}")
#             return None
    
#     def get_bq_model_id_by_version(self, display_name, version_id):
#         model_resource_name = f"projects/{self.project_id}/locations/{self.location}/models/{display_name}@{version_id}"
#         model_version = aiplatform.Model(model_resource_name)
#         model_dict = model_version.to_dict()

#         # Get version creation time
#         version_create_time = model_dict['versionCreateTime']
#         # Convert to datetime and format as YYYYMMDD_HHMMSS
#         create_datetime = datetime.datetime.fromisoformat(version_create_time.replace('Z', '+00:00'))
#         timestamp_str = create_datetime.strftime("%Y%m%d_%H%M%S")

#         if "boosted_tree" in display_name:
#             bq_model_id = f"boosted_tree_regressor_model_{timestamp_str}"
#         elif "matrix_factorization" in display_name:
#             bq_model_id = f"matrix_factorization_model_{timestamp_str}"
#         else:
#             raise ValueError(f"Model type for {display_name} not recognized.")

#         return bq_model_id

#     def get_model_from_registry(self, display_name: str) -> Optional[str]:
#         if "boosted_tree" in display_name:
#             return f"{self.project_id}.{self.dataset_id}.boosted_tree_regressor_model"
#         elif "matrix_factorization" in display_name:
#             return f"{self.project_id}.{self.dataset_id}.matrix_factorization_model"
#         else:
#             print(f"Model type for {display_name} not recognized.")
#             return None
    
#     def get_predictions(self, user_id):
#         """
#         Generate book recommendations for a given user_id using the selected model.
#         """
#         model_info = get_selected_model_info()
#         if not model_info:
#             raise ValueError("No model selected for predictions.")
        
#         model_name = model_info['display_name']

#         bq_model_id = self.get_model_from_registry(model_name)

#         if not bq_model_id:
#             raise ValueError(f"Could not retrieve BigQuery model ID for model {model_name}.")
#         if "matrix_factorization" in model_name:
#             predictions = self.get_mf_predictions(bq_model_id, user_id)
#         elif "boosted_tree" in model_name:
#             predictions = self.get_bt_predictions(bq_model_id, user_id)
#         else:
#             raise ValueError(f"Model type for {model_name} not recognized.")     
#         return predictions

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate book predictions for a user")
#     parser.add_argument("--user_id", type=str, required=True, help="User ID to generate predictions for")
    
#     args = parser.parse_args()
    
#     generator = GeneratePredictions()
#     predictions = generator.get_predictions(args.user_id)
#     print(predictions)

# # Sample runner command:
# # python -m src.generate_predictions --user_id "f4ce975e2b7f47212a0606ef7f103e00"


from src.model_deployment import get_selected_model_info
import os
from google.cloud import bigquery, aiplatform
from typing import Optional, List
import datetime
import argparse
import pandas as pd
from tqdm import tqdm
import json

class GeneratePredictions:
    def __init__(self):
        """
        Initialize BigQuery ML model training.
        """
        if os.environ.get("AIRFLOW_HOME"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("AIRFLOW_HOME")+"/gcp_credentials.json"

        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = "books"
        self.location = "us-central1"
        aiplatform.init(
            project=self.project_id,
            location=self.location
        )

    def get_all_users(self):
        """
        Returns the hardcoded list of user_ids provided by the user.
        """
        raw_users = [
            { "user_id_clean": "2faa2ef7e9062a7339ed1e4299c7ecaf" }, { "user_id_clean": "b7d51c3e72c2e202995f17e527c31ee2" },
            { "user_id_clean": "8fabca857b5c4b3c94bf1f2bf246dbe6" }, { "user_id_clean": "4ef42497be8b2e7e467c0db5bb6fb2e2" },
            { "user_id_clean": "8955df0f90cf9805f7489169c6185ac3" }, { "user_id_clean": "9c6e9a79ba00dd22e3798b253ef3abf3" },
            { "user_id_clean": "0d716d6e6f6df5df89e4038324c568c4" }, { "user_id_clean": "509eacb6699a5b5579bf6b204c25ad0b" },
            { "user_id_clean": "30f9c3f4711bd2e62b254e9fc0eb5f79" }, { "user_id_clean": "664e1e0191104537a75109ee47e6a20f" },
            { "user_id_clean": "1d51b5b54763b7a8766b0e1d9d427615" }, { "user_id_clean": "fa5309cde79f50ed27de291e3099c745" },
            { "user_id_clean": "4ceed8a6267ce08ce5c405c4e2fdf1c1" }, { "user_id_clean": "d32084057134357c38bd0c3a377fbef1" },
            { "user_id_clean": "4597ba0bb52054eae1e87534c78b13b8" }, { "user_id_clean": "40566972c328e87d4149345316a50249" },
            { "user_id_clean": "99f506284f34a191b20fac452470d76c" }, { "user_id_clean": "c36653d1becc6532f6f8cee8b60feecb" },
            { "user_id_clean": "96f6715475cac2516fe99183652be24c" }, { "user_id_clean": "087dc85710b6f04e0f4f4ab5aee1dd54" },
            { "user_id_clean": "77e23fade679f36143191defa1caa5d1" }, { "user_id_clean": "16d2e08e94a1639c3e62bcbf27010698" },
            { "user_id_clean": "fd2f28fc28927cba3fc3e47df062defd" }, { "user_id_clean": "e9cce797f5f7f4058b23309a28c3f834" },
            { "user_id_clean": "ff6f07733656515ac0b0a1d3a87701c7" }, { "user_id_clean": "1129be6ceacc39573e301e21aa6e94d3" },
            { "user_id_clean": "400266f139fab68196ee74382c5330d8" }, { "user_id_clean": "3bd95a28210264faa99d0f04b213e92d" },
            { "user_id_clean": "69f9d7bda6cd4993217762e60ca5d1ee" }, { "user_id_clean": "cbbc7e48ac3cc2e190e5066722d11163" },
            { "user_id_clean": "45812520dfd0ad1e03d440b53fd715ec" }, { "user_id_clean": "430ceaa4f6dbfce917074312985ee324" },
            { "user_id_clean": "04a2a05803eb8bd8c4e3e400d11c8a1f" }, { "user_id_clean": "34807a6205b94662cdc52cf30aa7259e" },
            { "user_id_clean": "6be9fa9b8edc7fe28d94b085d01346d3" }, { "user_id_clean": "d1955cb1b7fc558a50869f0e1e1d3065" },
            { "user_id_clean": "fa61a7c1306a8d31776d389ce48c6b6f" }, { "user_id_clean": "0c5628effcaceb51bd941485cd6b6854" },
            { "user_id_clean": "18f1dcf9a27ec2660b32f5cebcaec072" }, { "user_id_clean": "ea8fcc3adb695881791e99f8ad0ecfca" },
            { "user_id_clean": "3ee521eedb87ef771f364454db348a95" }, { "user_id_clean": "cdf46dd2e8a76cc6524beccd6a456537" },
            { "user_id_clean": "ea8a74f891f5a68749e0a5bb6d615b06" }, { "user_id_clean": "cefdd4dca30e73c190b484d668616ab7" },
            { "user_id_clean": "86b0f8caad0c89c9b9ee9c5061e7d3db" }, { "user_id_clean": "52121738cc67b091882beba9c1fa404b" },
            { "user_id_clean": "da15fc7c3ab623fbe50ff468ca83a7ea" }, { "user_id_clean": "21c6e116b77b50fea66f2caae07a11a4" },
            { "user_id_clean": "bd346bb71f5ad75d190ba1f93a77b1d9" }, { "user_id_clean": "6748b6dba2047068a381a096463e48ff" }
        ]
        # Extract just the string IDs from the dictionaries
        return [user['user_id_clean'] for user in raw_users]

    def get_mf_predictions(self, model_name, user_id):
        """Matrix Factorization prediction logic"""
        config = {
            "project_id": self.project_id,
            "dataset": self.dataset_id,
            "model_name": model_name
        }
        
        with open("src/mf_predictor_query.sql", "r") as file:
            query_template = file.read()
        
        query = query_template.format(**config)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        try:
            results = self.client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=False)
            if results.empty:
                 return pd.DataFrame(columns=['book_id', 'title', 'rating', 'author_names'])
            return results[['book_id', 'title', 'rating', 'author_names']]
        except Exception as e:
            print(f"Query failed for user {user_id}: {e}")
            return pd.DataFrame(columns=['book_id', 'title', 'rating', 'author_names'])

    def get_bt_predictions(self, model_name, user_id):
        """Boosted Tree prediction logic"""
        config = {
            "project_id": self.project_id,
            "dataset": self.dataset_id,
            "model_name": model_name
        }
        
        with open("src/bt_predictor_query.sql", "r") as file:
            query_template = file.read()
        
        query = query_template.format(**config)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        try:
            results = self.client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=False)
            if results.empty:
                return pd.DataFrame(columns=['book_id', 'title', 'rating', 'author_names'])
            return results[['book_id', 'title', 'rating', 'author_names']]
        except Exception as e:
            print(f"Query failed for user {user_id}: {e}")
            return pd.DataFrame(columns=['book_id', 'title', 'rating', 'author_names'])

    def get_model_from_registry(self, display_name: str) -> Optional[str]:
        if "boosted_tree" in display_name:
            return f"{self.project_id}.{self.dataset_id}.boosted_tree_regressor_model"
        elif "matrix_factorization" in display_name:
            return f"{self.project_id}.{self.dataset_id}.matrix_factorization_model"
        else:
            print(f"Model type for {display_name} not recognized.")
            return None

    def get_predictions(self, user_id):
        """
        Generate book recommendations for a given user_id using the selected model.
        """
        model_info = get_selected_model_info()
        if not model_info:
            raise ValueError("No model selected for predictions.")
        
        model_name = model_info['display_name']
        bq_model_id = self.get_model_from_registry(model_name)

        if not bq_model_id:
            raise ValueError(f"Could not retrieve BigQuery model ID for model {model_name}.")
            
        if "matrix_factorization" in model_name:
            predictions = self.get_mf_predictions(bq_model_id, user_id)
        elif "boosted_tree" in model_name:
            predictions = self.get_bt_predictions(bq_model_id, user_id)
        else:
            raise ValueError(f"Model type for {model_name} not recognized.")     
        
        return predictions

    def generate_bulk_predictions(self, output_file="recommendations.csv"):
        """
        Runs predictions for ALL users in the hardcoded list and saves to a CSV.
        """
        print("Loading user list...")
        all_users = self.get_all_users()
        print(f"Found {len(all_users)} users. Starting predictions...")

        results_list = []

        # Tqdm creates a progress bar in your terminal
        for user_id in tqdm(all_users):
            try:
                # Get predictions for this user
                preds_df = self.get_predictions(user_id)
                
                if not preds_df.empty:
                    # Collect all book IDs into a list
                    book_ids = preds_df['book_id'].tolist()
                    
                    # Append to our results
                    results_list.append({
                        "user_id": user_id,
                        "recommended_book_ids": book_ids 
                    })
                else:
                    # Optional: Record users with no recommendations as empty lists
                    results_list.append({
                        "user_id": user_id,
                        "recommended_book_ids": [] 
                    })

            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                continue

        # Convert to DataFrame
        final_df = pd.DataFrame(results_list)
        
        # Save to CSV
        final_df.to_csv(output_file, index=False)
        print(f"✅ Successfully saved recommendations for {len(final_df)} users to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate book predictions")
    parser.add_argument("--mode", type=str, choices=['single', 'bulk'], default='single', 
                        help="Run for a 'single' user or 'bulk' for all users in the hardcoded list")
    parser.add_argument("--user_id", type=str, help="User ID (required if mode is single)")
    
    args = parser.parse_args()
    
    generator = GeneratePredictions()

    if args.mode == 'bulk':
        generator.generate_bulk_predictions()
    else:
        if not args.user_id:
            print("Error: --user_id is required for single mode.")
        else:
            predictions = generator.get_predictions(args.user_id)
            print(predictions)