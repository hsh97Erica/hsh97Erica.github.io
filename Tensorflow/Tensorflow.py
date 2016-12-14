import tensorflow as tf #관련 라이브러리 추가
from tensorflow.examples.tutorials.mnist import input_data #관련 패키지 추가

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True) #이미지 데이터를 로드합니다

# Set up model
x = tf.placeholder(tf.float32, [None, 784]) # 하나의 이미지는 28X28=784이다 , None은 배치 크기에 해당되는데 어떤 크기도 될 수 있다는 의미이며 placeholder는 입력값의 형태를 미리 만들어 둔다
W = tf.Variable(tf.zeros([784, 10])) # w을 0으로 초기화 W는 784x10 행렬 (우리가 784개의 입력 특징(이미지의 픽셀수)과 10개의 출력이 있으므로) 이다
b = tf.Variable(tf.zeros([10])) # 0에서 9까지의 인지 데이터를 생성합니다(10개의 클래스로 이루어진 차원 배열)
y = tf.nn.softmax(tf.matmul(x, W) + b) #벡터화된 입력 이미지 x를 가중치 행렬 W로 곱하고, 편향(bias) b를 더한 다음 각 클래스에 지정된 소프트맥스 확률들을 계산합니다.

y_ = tf.placeholder(tf.float32, [None, 10])#목표 출력 클래스 y_ 또한 2d 텐서인데, 각 행은 대응되는 MNIST 이미지가 어떤 숫자 클래스에 속하는지를 나타내는 one-hot 10차원 벡터입니다

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) #텐서의 차원을 가로질러 각 요소들의 합을 구합니다
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) #학습 비율 0.01을 인자로 경사 하강법을 적용하여 크로스 엔트로피를 최소화 하도록 합니다

# Session
init = tf.initialize_all_variables() #변수들의 배열을 초기화한뒤 그것들을 포함한 op객체를 반환합니다

sess = tf.Session() #텐서플로우의 그래프 계산을 도와주는 객체를 생성합니다
sess.run(init)#Variables가 세션 안에서 사용되기 전에 반드시 그 세션을 사용하여 초기화되어야 합니다. 이 단계는 이미 지정되어 있는 초기값을 (이 경우 0으로 채워진 텐서) 사용하고, 각 Variable에 할당해야 합니다. 이 과정은 모든 Variables 에 대하여 한 번에 수행할 수 있습니다.

# Learning
for i in range(1000): #학습을 반복한다(이경우 1000번 학습한다)
  batch_xs, batch_ys = mnist.train.next_batch(100) #학습 데이터 셋에서 무작위로 선택된 100개의 데이터로 구성된 배치를 가져옵니다
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})#placeholder의 자리에 데이터를 넣을 수 있도록 train_step을 실행하여 배치 데이터를 넘긴후 확률적 경하 사겅봅을 사용해 무작위 데이터의 작은 배치를 사용해 학습을 시킵니다

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)#리턴형은 boolean으로 이루어진 리스트이며 모델이 라벨을 올바르게 예측했는지 확인해보는데 tf.argmax(y,1)는 우리의 모델이 생각하기에 각 데이터에 가장 적합하다고 판단한(가장 증거값이 큰) 라벨이며 tf.argmax(y_,1)는 실제 라벨이다, 이때 우리는 tf.equal을 사용하여 우리의 예측이 맞았는지 확인할 수 있다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#correct_prediction의 값으로 얼마나 많이 맞았는지 판단하려면, 이 값을 부동소수점 값으로 변환한 후 평균을 계산하면 된다

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))#우리의 테스트 데이터를 대상으로 정확도를 계산해 출력합니다
