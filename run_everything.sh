#!/bin/sh

bash prepare_data.sh

bash run_all_instance-based_limited_data.sh
bash run_all_instance-based_full_data.sh

bash run_all_similarity-based.sh