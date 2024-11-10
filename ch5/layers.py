import numpy as np

class MulLayer:
    def __init__(self) -> None:
        # 直接pass的话，必须forward才有这个属性
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        return x*y
    
    def backward(self,dout):
        return self.y*dout,self.x*dout

class AddLayer():
    def __init__(self) -> None:
        pass

    def forward(self,x,y):
        return x+y
    
    def backward(self,out):
        return out,out

if __name__ == '__main__':
    layer_apple = MulLayer()
    out1 = layer_apple.forward(100,2)

    layer_tax = MulLayer()
    out2 = layer_tax.forward(out1,1.1)

    print('output of layer 1 = ',out1,
        'output of layer 2 = ',out2)

    dx2,dy2 = layer_tax.backward(1)
    dx1,dy1 = layer_apple.backward(dx2)

    print(dx1,dy1,dx2,dy2)
    # 初始化输入参数
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    # 定义计算层
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()


    # 前向传播
    apple_price = mul_apple_layer.forward(apple, apple_num)        # (1)
    orange_price = mul_orange_layer.forward(orange, orange_num)    # (2)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
    price = mul_tax_layer.forward(all_price, tax)                  # (4)

    # 反向传播
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)              # (4)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price) # (3)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price) # (2)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)    # (1)

    # 输出结果
    print("Price:", price)  # 715
    print("Gradients:", dapple_num, dapple, dorange, dorange_num, dtax)  # 110 2.2 3.3 165 650

