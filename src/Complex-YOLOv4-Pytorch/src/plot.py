import matplotlib.pyplot as plt
import numpy as np

def box_plot(data_a, data_b, data_c):
    # 1. 기본 스타일 설정
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (4, 3)
    plt.rcParams['font.size'] = 12

    # 2. 데이터 준비
    np.random.seed(0)
    data_a = np.random.normal(0, 2.0, 1000)
    data_b = np.random.normal(-3.0, 1.5, 500)
    data_c = np.random.normal(1.2, 1.5, 1500)

    # 3. 그래프 그리기
    fig, ax = plt.subplots()

    ax.boxplot([data_a, data_b, data_c], whis=2.5)
    plt.xticks([1, 2, 3], ['mon', 'tue', 'wed'])
    ax.set_ylim(-10.0, 10.0)
    ax.set_xlabel('Data Type')
    ax.set_ylabel('Value')

    plt.show()

box_plot()