import os
import matplotlib.pyplot as plt
import task_3
import task_2_model_2

if __name__ == "__main__":

    #training the model from task 3
    trainer_resNet = task_3.Trainer()
    trainer_resNet.train()

    #traingin out best model from task 2
    trainer_model_2 = task_2_model_2.Trainer()
    trainer_model_2.train()

    os.makedirs("plots", exist_ok=True)

    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer_resNet.VALIDATION_LOSS, label="Validation loss ResNet")
    plt.plot(trainer_resNet.TRAIN_LOSS, label="Training loss ResNet")
    plt.plot(trainer_resNet.TEST_LOSS, label="Testing loss ResNet")
    plt.plot(trainer_model_2.VALIDATION_LOSS, label="Validation loss Model 2")
    plt.plot(trainer_model_2.TRAIN_LOSS, label="Training loss Model 2")
    plt.plot(trainer_model_2.TEST_LOSS, label="Testing Loss Model 2")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(trainer_resNet.VALIDATION_ACC, label="Validation accuracy ResNet")
    plt.plot(trainer_resNet.TRAIN_ACC, label="Training accuracy ResNet")
    plt.plot(trainer_resNet.TEST_ACC, label="Testing accuracy ResNet")
    plt.plot(trainer_model_2.VALIDATION_ACC, label="Validation accuracy Model 2")
    plt.plot(trainer_model_2.TRAIN_ACC, label="Training accuracy Model 2")
    plt.plot(trainer_model_2.TEST_ACC, label="Testing accuracy Model 2")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy.png"))
    plt.show()
