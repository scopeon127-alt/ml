def initiate_data_transformation(self, train_path, test_path):

    try:
        # =========================================
        # Read train and test CSV files
        # =========================================
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info("Read train and test data completed")


        # =========================================
        # Get preprocessing object
        # =========================================
        preprocessing_obj = self.get_data_transformer_object()


        # =========================================
        # Define target column
        # =========================================
        target_column_name = "math score"


        # =========================================
        # Separate input and target features
        # =========================================
        input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
        target_feature_test_df = test_df[target_column_name]


        logging.info("Applying preprocessing object")


        # =========================================
        # IMPORTANT:
        # Fit only on TRAIN
        # Transform both train and test
        # =========================================
        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


        # =========================================
        # Combine input + target
        # =========================================
        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


        # =========================================
        # Save preprocessing object (for future use)
        # =========================================
        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
        )

        logging.info("Preprocessing object saved successfully")


        # =========================================
        # Return arrays
        # =========================================
        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
        )


    except Exception as e:
        raise CustomException(e, sys)
