1. tar –xvf  CUB_200_2011.tar

2. put filenames_train.pickle, filenames_test.pickle and captions.pickle in the CUB_200_2011 file

3. run python preprocess_bird.py
   Before running it, we need to change the "abspath" value, which is the absolute path of the dataset.

   >It performs the functions including:
    1. process_test_text()

    2. process_train_text()

    3. resize_test_image()

    4. resize_train_image()