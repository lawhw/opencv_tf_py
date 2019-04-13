# -*- coding: utf-8 -*-


from PIL import Image
import struct


def read_image(filename):
    f = open(filename, 'rb')


    index = 0
    buf = f.read()


    f.close()


    magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')


    for i in range(images):
        image = Image.new('L', (columns, rows))


        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')


                print('save ' + str(i) + 'image')
                image.save('./tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte/' + str(i) + '.png')


def read_label(filename, saveFilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()


    f.close()


    magic, labels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    labelArr = [0] * labels
    #labelArr = [0] * 2000


    for x in range(labels):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')


    save = open(saveFilename, 'w')


    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')


    save.close()
    print('save labels success')


if __name__ == '__main__':
    read_image('./tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
    read_label('./tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte', './tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte/label.txt')

    read_image('./tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte/train-images.idx3-ubyte')
    read_label('./tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte/train-labels.idx1-ubyte',
               './tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte/label.txt')
