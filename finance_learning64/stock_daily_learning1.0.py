from datetime import datetime
import sqlite3 as sql
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


# configuration
#                               O * W + b -> 3 labels for each time series, O[? 7], W[7 3], B[3]
#                               ^ (O: output 7 vec from 7 vec input)
#                               |
#              +-+  +-+        +--+
#              |1|->|2|-> ...  |60| time_step_size = 60
#              +-+  +-+        +--+
#               ^    ^    ...   ^
#               |    |          |
# time series1:[7]  [7]   ...  [7]
# time series2:[7]  [7]   ...  [7]
# time series3:[7]  [7]   ...  [7]
# ...
# time series(250) or time series(750) (batch_size 250 or test_size 1000 - 250)
#      each input size = input_vec_size=lstm_size=7

# configuration variables
input_vec_size = lstm_size = 7
time_step_size = 10 # 60
label_size = 3
evaluate_size = 3
lstm_depth = 4

total_size = 200#000   //토탈 사이즈  59000
batch_size = 150 # 000    //한번에 처리하는 훈련데이타
test_size = total_size - batch_size    # //500 = 2000 -1500 검증 데이타    (test/batch)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))  #정규 분포로부터 랜덤 값을 출력

def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR shape: (time_step_size * batch_size, input_vec_size)
    X_split = tf.split(axis=0, num_or_size_splits=time_step_size, value=XR) # split them to time_step_size (60 arrays)

    # Each array shape: (batch_size, input_vec_size)

    # Make lstm with lstm_size (each input vector size)
    cell = rnn.LSTMCell(lstm_size)
    cell = rnn.DropoutWrapper(cell = cell, output_keep_prob = 0.5)
    cell = rnn.MultiRNNCell([cell] * lstm_depth, state_is_tuple = True)

    #lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size)
   # train_outputs, train_states = tf.contrib.rnn.static_rnn(lstm_cell, train_x_temp, dtype=tf.float32)
    #train_output = tf.matmul(train_outputs[-1], layer['weights']) + layer['biases']



    # Get lstm cell output, time_step_size (60) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = rnn.static_rnn(cell, X_split, dtype=tf.float32)

   # train_outputs, train_states = tf.contrib.rnn.static_rnn(lstm_cell, train_x_temp, dtype=tf.float32)

   # +        lstm_cell = tf.contributes.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, cell.state_size # State size to initialize the stat

def read_series_datas(conn, code_dates):
    X = list()
    Y = list()

    for code_date in code_dates:
        cursor = conn.cursor()
        cursor.execute("SELECT open, high, low, close, volume, hold_foreign, st_purchase_inst FROM stock_daily_series WHERE code = '{0}' AND date >= '{1}' ORDER BY date LIMIT {2}".format(code_date[0], code_date[1], time_step_size + evaluate_size))
        items = cursor.fetchall()

        X.append(np.array(items[:time_step_size]))

        price = items[-(evaluate_size + 1)][3]
        max = items[-evaluate_size][1]
        min = items[-evaluate_size][2]

        for item in items[-evaluate_size + 1:]:
            if max < item[1]:
                max = item[1]
            if item[2] < min:
                min = item[2]

        if (min - price) / price < -0.02:  #2% s내린갑
            Y.append((0., 0., 1.))
        elif (max - price) / price > 0.04:  #4% 증가 된값
            Y.append((1., 0., 0.))
        else:
            Y.append((0., 1., 0.))

    arrX = np.array(X)
    norX = (arrX - np.mean(arrX, axis = 0)) / np.std(arrX, axis = 0)  # 입력값의 - 열평균/표준편차
    return norX, np.array(Y)

def read_datas():
    conn = sql.connect("./finance_learning.db")
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT code FROM stock_daily_series order by date LIMIT 100")
        codes = cursor.fetchall()

        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT date FROM stock_daily_series ORDER BY date  LIMIT 10000")
        dates = cursor.fetchall()[:-(time_step_size + evaluate_size)]

        cnt = total_size
        code_dates = list()
        for date in dates:
            for code in codes:
                code_dates.append((code[0], date[0]))
                if --cnt <= 0:
                    break
            if --cnt <= 0:
                break

        np.random.seed()
        np.random.shuffle(code_dates)

        trX = list()
        trY = list()
        trX, trY = read_series_datas(conn, code_dates[:batch_size])
        teX, teY = read_series_datas(conn, code_dates[-test_size:])

    return trX, trY, teX, teY

trX, trY, teX, teY = read_datas()
# trX  (15000,60,7)
#tr V (15000 3)
#tex (45000 60,7)
#tey(45000,3)
X = tf.placeholder(tf.float32, [None, time_step_size, input_vec_size])
Y = tf.placeholder(tf.float32, [None, label_size])

# get lstm_size and output 3 labels
W = init_weights([lstm_size, label_size])
B = init_weights([label_size])

py_x, state_size = model(X, W, B, lstm_size)
#py_x (?,3)
#state_size c=7 h=7
loss = tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)
cost = tf.reduce_mean(loss)
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(1000):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            t_lo,_ = sess.run([loss,train_op], feed_dict={X: trX[start:end], Y: trY[start:end]})
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        test_indices = np.arange(len(teX))  # Get A Test(test_indices) Batch
        #np.random.shuffle
        test_indices = test_indices[0:test_size]

        org = teY[test_indices]
        result = sess.run(predict_op, feed_dict={X: teX[test_indices], Y: teY[test_indices]})

        print(i, 'loss:',result,np.mean(np.argmax(org, axis=1) == result))
