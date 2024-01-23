import matplotlib.pyplot as plt

def openreadtxt(file_name):
    with open(file_name,'r', encoding='utf-8') as f:  
        file_data = f.readlines() 
    
    train_metric = {
        'loss': [],
        'psnr': [],
        'ssim': []
    }
    test_metric = {
        'loss': [],
        'psnr': [],
        'ssim': []
    }

    for row in file_data:
        if 'Train, loss is:' in row:
            tmp  = row.split(',')[-3:]
            train_metric['loss'].append(float(tmp[0].split(':')[-1]))
            train_metric['psnr'].append(float(tmp[1].split('is')[-1]))
            train_metric['ssim'].append(float(tmp[2].split('is')[-1]))

        if 'testsets is' in row:
            tmp  = row.split(',')[-3:]
            # test_metric['loss'].append(float(tmp[0].split(':')[-1]))
            test_metric['psnr'].append(float(tmp[1].split('is')[-1]))
            test_metric['ssim'].append(float(tmp[2].split('is')[-1]))
    # print(test_metric)

    return train_metric, test_metric
  
  
if __name__=="__main__":
    PATH = [r'D:\workspace\experiments\UnsupervisedLFSR\log\SR_8x8_4x_HASH_5120\ALL\hash_v4_res\hash_v4_res.txt', 
            # r''
            ]
    
    # plt.style.use('ggplot')
    fig=plt.figure(1)
    # plt.xticks(fontname="Epoch", fontsize=10)
    # plt.yticks(fontname="PSNR (dB)", fontsize=10)

    # plt.legend(["train", "test"], ncol=2,
    #         prop={"family": "Times New Roman", "size": 20})

    plt.xlabel('Epoch', fontdict={"family": "Times New Roman", "size": 10})
    plt.ylabel('Error', fontdict={"family": "Times New Roman", "size": 10})

    for path in PATH:
        train_metric, test_metric = openreadtxt(path)   
        plt.plot(train_metric['loss'], color='r', alpha=0.8, linewidth=1, label='Train')        
        # plt.plot(train_metric['psnr'], color='r', alpha=0.8, linewidth=1, label='Train')
        # plt.plot(test_metric['psnr'], color='g', alpha=0.8, linewidth=1, label='Test')
    plt.legend()
    plt.show()