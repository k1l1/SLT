from torchvision import transforms as tf

tf_cifar_train = tf.Compose([
    tf.RandomCrop(32, padding=4),
    tf.RandomHorizontalFlip(),
    tf.ToTensor(),
    tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

tf_cifar_test = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])