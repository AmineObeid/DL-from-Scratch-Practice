# import numpy as np
# from tensor import Tensor

# def add(x: Tensor, y: Tensor) -> Tensor:
#     x_data = x.data
#     y_data = y.data

#     data = np.add(x_data, y_data)
#     parents = [x, y]
#     op = "add"

#     result = Tensor(data=data, parents=parents, op=op)

#     return result

# def mul(x: Tensor, y: Tensor) -> Tensor:
#     """Element wise multiplication"""
#     x_data = x.data
#     y_data = y.data

#     data = np.multiply(x_data, y_data)
#     parents = [x, y]
#     op = "mul"

#     result = Tensor(data=data, parents=parents, op=op)

#     return result

# def neg(x: Tensor) -> Tensor:
#     x_data = x.data

#     data = np.negative(x_data)
#     parents = [x]
#     op = "neg"

#     result = Tensor(data=data, parents=parents, op=op)

#     return result

# def sub(x: Tensor, y: Tensor) -> Tensor:
#     return add(x, neg(y))

# def matmul(x: Tensor, y:Tensor) -> Tensor:
#     """Matrix Multiplication"""
#     x_data = x.data
#     y_data = y.data

#     data = np.matmul(x_data, y_data)
#     parents = [x, y]
#     op = "matmul"

#     result = Tensor(data=data, parents=parents, op=op)

#     return result

# #TODO: Sum, Mean, Reshape ...