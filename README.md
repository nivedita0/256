## Data cleaning to improve the performance of the recommender system.

Strategy - In order to address the cold-start items concern, a data cleaning approach was evaluated where the items (1529 in this case) identified as cold-items in the EDA stage were dropped. An additional condition was added to drop a user if the number of items it interacted with after dropping the cold items was 0 (which in our case didn’t happen). However, this dropped the metrics slightly. This could be because even low-frequency items still contribute meaningfully.
