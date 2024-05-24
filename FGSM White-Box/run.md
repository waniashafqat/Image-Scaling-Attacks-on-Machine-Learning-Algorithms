To generate adversarial examples, run:

python fgsm.py "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset/lol.keras" "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset" gen_data -s 30 30 -e 0.01 -p "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset/results/"


To check model robustness, run:

python fgsm.py "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset/lol.keras" "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset" check_loss -s 30 30 -e 0.01 -p "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset/results/"


To train the model to defend against FGSM attacks, run:

python fgsm.py "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset/lol.keras" "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset" train -s 30 30 -p "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset/results/" -b 16 -n 15 -v 10

To perform all tasks sequentially, run:

python fgsm.py "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset/lol.keras" "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset" all -s 30 30 -e 0.01 -p "C:/Users/wania/OneDrive/Desktop/GTSRB Dataset/results/" -b 16 -n 15 -v 10

![image](https://github.com/waniashafqat/Image-Scaling-Attack-on-Machine-Learning-Algorithms/assets/73712563/5091ca50-7408-4c16-ac2a-6fc70bc89dae)
![image](https://github.com/waniashafqat/Image-Scaling-Attack-on-Machine-Learning-Algorithms/assets/73712563/e8162d23-7887-4e54-a65a-36c489337837)
![image](https://github.com/waniashafqat/Image-Scaling-Attack-on-Machine-Learning-Algorithms/assets/73712563/d9624912-3d0a-44db-8085-5b097f7cafee)
