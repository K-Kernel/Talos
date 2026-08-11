# The tales of Talos
A transformer inference engine in C++/CUDA, built from scratch. No PyTorch , no BLAS every operation implemented by hand as a learning project.

## Status
- [x] Tensor(flat, row-major)
- [x] Matrix multiplication, matrix sum and Sigmoid Linear unit 
- [x] Transpose , RSMNorm 
- [x] Weight loading 
- [x] Tokenizer
- [x] forward pass -> text generation
- [x] KV cache
- [ ] quantization, CUDA

## Build

```
    cd build 
    make clean
    make run
```


## Logs
(link to my website or maybe some resource i use)

- Fri 7 Aug: I finished the foward , and print once the first word of my transformer, the final step was reading the tokenizer binary file to decode the exact word it predict, the file is divided into a header that contain the maximum size that a string can take , a float ( that i ignor), a len and the bytes of the string , what i do its going thorugh the binary file reading the len and then reading the strign and writing direct into the string buffer , rememeber that string are jsust a list of character connected by points .

- Tue 11 Aug: Some fucntion like SiLU, softmax and transpose could be done diretcly into the Tensor but they retrun so its possible to chain operations.

## Docs

### Build/
This is the file where i run the make to build and run my cpp code using the command ``` make clean ``` and ```make run ```. Everytime I change the CMakeList.txt I need to run ``` cmake --build build ``` to rebuild everything.

### main.cpp
This file is the main entry point , it's the file that is run when i do ```make run ``` , this file is where i do most of the test and write the code before extracting it into functions.

### ops.cpp
This file is where most of the function live. It has 12 function:
- matmul: this function takes 2 tensors and return a new tensor. It perform matrix multiplicaiton usign the naive implementation O(n^3)

- matadd_elementwise: take two tensors. It iterate through the data buffer and add the elements of one into the other. Return a a result tensor 

- SiLU: it takes a tensor. It iterate through the data dividing each x to 1 plus the euler(e) to the exponent of x. It returns a new tenosr with the values modified

- softmax: It takes one tensor. It goes through each row , pick the largest number of that row, calculate the sum fo the exponents minus the sum ( so it doesn't overflow). Then it takes x minus teh biggest number ove the sum of exponents. This happens inside the vectore meaning it return nothing , using the reference of the tensor being passed.


- Transpose: It takes one tensor. It transpose one tensor , transpose means changing the rows with the correspoding columns. Then returns the result , it does not change the tensor 

- rmsnorm: takes a tensor and a tensor of weight. It then apply root mean square layer normalisation into the tensor directly changing the results




