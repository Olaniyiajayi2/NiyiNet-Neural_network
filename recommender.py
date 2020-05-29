class ImageRecommender : 
    
    def __init__(self, model, list_of_image, filespath) : 
        self.model = model
        self.filespath = filespath
        self.list_of_image = list_of_image
        #since ouput.shape return object dimension just eval it to get integer ...
        self.image_width = eval(str(self.model.layers[0].output.shape[1]))
        self.image_height = eval(str(self.model.layers[0].output.shape[2]))
        # remove the last layers in order to get features instead of predictions
        self.image_features_extractor = Model(inputs=self.model.input, 
                                              outputs=self.model.layers[-2].output)
        self.processed_image = self.Pics2Matrix()
        self.sim_table = self.GetSimilarity(self.processed_image)
        
    def ddl_images(self, image_url) :
        try : 
            return load_img(self.filespath + image_url, 
                            target_size=(self.image_width, self.image_height))
        except OSError : 
            # image unreadable // remove from list
            self.list_of_image = [x for x in self.list_of_image if x != image_url]
            #self.list_of_image.remove(image_url)
            pass
        
    def Pics2Matrix(self) :
        """
        # convert the PIL image to a numpy array
        # in PIL - image is in (width, height, channel)
        # in Numpy - image is in (height, width, channel)
        # convert the image / images into batch format
        # expand_dims will add an extra dimension to the data at a particular axis
        # we want the input matrix to the network to be of the form (batchsize, height, width, channels)
        # thus we add the extra dimension to the axis 0.
        """
        #from keras.preprocessing.image import load_img,img_to_array
        list_of_expanded_array = list()
        for i in tqdm(range(len(self.list_of_image) - 1)) :
            try :
                tmp = img_to_array(self.ddl_images(self.list_of_image[i]))
                expand = np.expand_dims(tmp, axis = 0)
                list_of_expanded_array.append(expand)
            except ValueError : 
                self.list_of_image = [x for x in self.list_of_image if x != self.list_of_image[i]]
                #self.list_of_image.remove(self.list_of_image[i])
        images = np.vstack(list_of_expanded_array)
        """
        list_of_expanded_array = [try np.expand_dims(img_to_array(self.ddl_images(self.list_of_image[i])), axis = 0) except ValueError pass \
                                  for i in tqdm(range(len(self.list_of_image)))]
        images = np.vstack(list_of_expanded_array)
        #from keras.applications.imagenet_utils import preprocess_input()
        # prepare the image for the  model"
        """
        return preprocess_input(images)
    
    def GetSimilarity(self, processed_imgs) :
        print('============ algorithm predict featurs =========')
        imgs_features = self.image_features_extractor.predict(processed_imgs)
        print("Our image has %i features:" %imgs_features.size)
        cosSimilarities = cosine_similarity(imgs_features)
        cos_similarities_df = pd.DataFrame(cosSimilarities, 
                                           columns=self.list_of_image[:len(self.list_of_image) -1],
                                           index=self.list_of_image[:len(self.list_of_image) -1])
        return cos_similarities_df
    
    def most_similar_to(self, given_img, nb_closest_images = 5):

        print("-----------------------------------------------------------------------")
        print("original manga:")

        original = self.ddl_images(given_img)
        plt.imshow(original)
        plt.show()

        print("-----------------------------------------------------------------------")
        print("most similar manga:")

        closest_imgs = self.sim_table[given_img].sort_values(ascending=False)[1:nb_closest_images+1].index
        closest_imgs_scores = self.sim_table[given_img].sort_values(ascending=False)[1:nb_closest_images+1]

        for i in range(0,len(closest_imgs)):
            original = self.ddl_images(closest_imgs[i])
            plt.imshow(original)
            plt.show()
            print("similarity score : ",closest_imgs_scores[i])
