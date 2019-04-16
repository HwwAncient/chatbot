import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request

def test(params, infos):

    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow

    x_data, _ = pickle.load(open('chatbot.pkl', 'rb'))
    ws = pickle.load(open('ws.pkl', 'rb'))

    for x in x_data[:5]:
        print(' '.join(x))

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    # 加载训练好的模型
    save_path = './model/s2ss_chatbot.ckpt'

    tf.reset_default_graph()

    # 不用beamsearch, beam_width设置为0, 使用贪心搜索
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=0,
        **params
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            # user_text = input('请输入您的句子:')
            # if user_text in ('exit', 'quit'):
            #     exit(0)
            # x_test = [list(user_text.lower())]
            x_test = [list(infos.lower())]
            # TODO [modify] 源代码中此处并未加add_end=False而是用了默认的True, 即使是预测模式我认为encoder的输入不应该有<END>
            # TODO 预测模式这里也只用输入x而不用输入y
            bar = batch_flow([x_test], ws, 1, add_end=False)
            x, xl = next(bar)
            x = np.flip(x, axis=1)

            print(x, xl)

            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(pred)

            print(ws.inverse_transform(x[0]))

            for p in pred:
                ans = ws.inverse_transform(p)
                print(ans)
                # 就一个预测句子, 故可直接返回
                return ans


app = Flask(__name__)

@app.route('/api/chatbot', methods=['get'])
def chatbot():
    # import json
    # test(json.load(open('params.json')))

    # TODO 现在每次访问该接口都需要构建图, 导致每次产生回答的时间都比较久

    # args的名称只要和http方法中请求的参数名称相同即可, 如此处的'info'
    # 注意输入URL的参数不应带引号
    # eg: 不用info='abc'而用info=abc
    infos = request.args['infos']

    import json
    text = test(json.load(open('params.json')), infos)

    # 不能直接返回text, 因为它是一个列表格式不能在前端网页展示
    # 虽然输出的列表中有</s>, 但是浏览器会把它当做一个未闭合的标签, 不进行显示
    # eg：text = ['我','想','你','</s>'], 但join后浏览器页面还是只显示"我想你"(没有双引号)
    # TODO 尽管浏览器不显示, 但其他设备可能显示, 最好手动去掉</s>
    return "".join(text)


if __name__ == '__main__':
    # 自动检测程序代码是否改动, 如若改动则重启app.run(), 代码改动完了之后还需要CRTL+S保存才能检测的到
    app.debug = True

    # 端口默认是127.0.0.1只是试用于本机调试不连接网络, 设置为0.0.0.0可以使程序所在的服务器被外部访问
    app.run(host='0.0.0.0', port=8000)
