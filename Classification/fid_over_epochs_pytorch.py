from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from scipy.linalg import sqrtm
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda'
with tf.device('/gpu:0'): 
    #class fid_calculate()
    import numpy as np
    import os
    IMG_SIZE = 128

    def load_model(model_path):
        return tf.saved_model.load(model_path)

    def generate_noise(num_samples, noise_dim):
        return tf.random.normal([num_samples, noise_dim])

    def save_images(images, folder_path,add_to_folder):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        images_uint8 = tf.image.convert_image_dtype(images, tf.uint8, saturate=True)
        
        for i, img in enumerate(images_uint8):
            # Generate a new filename if the file already exists
            file_path = os.path.join(folder_path, f'generated_image_{i}.jpeg')
            if(add_to_folder == 1):
                while os.path.exists(file_path):
                    i += 1  # Increment the counter if file exists
                    file_path = os.path.join(folder_path, f'generated_image_{i}.jpeg')  # Update file path

            encoded_img = tf.image.encode_jpeg(img)
            tf.io.write_file(file_path, encoded_img)

    def generate_imgs(model_path, num_images, noise_dim):
        # Load the model
        loaded_model = load_model(model_path)
        print(loaded_model.signatures['serving_default'])
        generator_signature = loaded_model.signatures['serving_default']
        
        noise = generate_noise(num_images, noise_dim)
        if(IMG_SIZE == 64):
            generated_images = generator_signature(dense_2_input=noise)['conv2d_transpose_7']
        elif(IMG_SIZE == 128):
            generated_images = generator_signature(dense_2_input=noise)['conv2d_transpose_9']
        
        images = (generated_images.numpy() + 1.0) / 2.0
        
        return(images)

        # Save images
        # save_images(images, output_folder,add_to_folder)
        

    def compute_fid_pytorch(fake_images, real_images):
        # Ensure images are on the same device and in PyTorch tensor format
        fake_images = torch.tensor(fake_images).float().to('cuda:0')
        real_images = torch.tensor(real_images).float().to('cuda:0')
        
        # Flatten the images
        num_fake_images = fake_images.size(0)
        fake_images_flat = fake_images.view(num_fake_images, -1).cpu().numpy()  # Move to CPU and convert to numpy
        
        num_real_images = real_images.size(0)
        real_images_flat = real_images.view(num_real_images, -1).cpu().numpy()  # Move to CPU and convert to numpy
        
        # Compute mean
        fake_mu = np.mean(fake_images_flat, axis=0)
        real_mu = np.mean(real_images_flat, axis=0)
        
        # Compute covariance using numpy
        fake_cov = np.cov(fake_images_flat, rowvar=False)
        real_cov = np.cov(real_images_flat, rowvar=False)
        
        # Computing the square root of product of covariances using scipy (since we're working with numpy now)
        from scipy.linalg import sqrtm
        covsqrt = sqrtm(fake_cov.dot(real_cov))
        
        # Ensure it's purely real if it's complex
        if np.iscomplexobj(covsqrt):
            covsqrt = covsqrt.real
        
        # FID score computation
        fid_score = np.sum((fake_mu - real_mu)**2) + np.trace(fake_cov + real_cov - 2 * covsqrt)
        
        return fid_score


    def load_real_images(dir):
        tumor_training_images = []
        for filename in os.listdir(dir):
            if filename.endswith('.jpg'):  # Assuming images are in JPG format
                image_path = os.path.join(dir, filename)
                image = plt.imread(image_path, format='jpg')
                tumor_training_images.append(image)

        # Convert the list of images to a NumPy array
        return np.asarray(tumor_training_images)
        
    #v38 is 128x128 images

    real_images_path = '/data/user1/AlphaGANMRI/cleaned128by128/Training/onlytumors'
    model_path = '/data/user1/AlphaGANMRI/v3/AlphaGAN/mri/alpha-d1.0-g5.0/v11/models/generator300'
    num_images = 4000  #  number of images to generate
    noise_dim = IMG_SIZE*IMG_SIZE  # dimension of the noise vector
    output_folder = '/data/user1/AlphaGANMRI/v3/generated_imgs/128x128testOnlyTumors'  # Specify the folder to save to


    fake_images = generate_imgs(model_path, num_images, noise_dim)
    real_images = load_real_images(real_images_path)
    # expand the dim to (4117, 128, 128, 1) from (4117, 128, 128)
    real_images = np.expand_dims(real_images, axis=-1)
    real_images = real_images/255
    real_images = real_images[:num_images]

    print(np.min(real_images))
    print(np.max(real_images))
    print(np.min(fake_images))
    print(np.max(fake_images))
    #fake_images = real_images
    #real_images = fake_images
    
    
    fid_score = compute_fid_pytorch(fake_images,real_images)
    
    print(fid_score)
    