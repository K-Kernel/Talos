# The tales of Talos
A transformer inference engine in C++/CUDA, built from scratch. No PyTorch , no BLAS every operation implemented by hand as a learning project.

## Status
- [x] Tensor(flat, row-major)
- [x] Matrix multiplication, matrix sum and Sigmoid Linear unit 
- [x] Transpose , RSMNorm 
- [x] Weight loading 
- [x] Tokenizer
- [x] forward pass -> text generation
- [ ] KV cache , quantization, CUDA

## Build

```
    cd build 
    make clean
    make run
```


## Logs
(link to my website or maybe some resource i use)

- Thu 30 jul: realised that there is a problme between how multipliaciton works and how the split fucntion I did work , the split funciton returend and array of an array but multiplication needs a tensor , specifically a flat buffer , so i need to find a way to split data inside the flat buffer, the problem is that rope only takes vector and position therefore i need to split the vecot after some back and forth thinking and evaluationg and fucntion that separate , make operation then agregate but it was discarded because to many opeeration ( althoguh getting work is the main priority, not optimising ) it came across of using iterator isntead of vector in rope , that way could just pass the iterator modify them inside the flat buffer. Also I spend way to much time tracing a bug in split funciton ( I was accessing a memory way past i should) and end up not even using that function. 

- Fri 7 Aug: I finished the foward , and print once the first word of my transformer, the final step was reading the tokenizer binary file to decode the exact word it predict, the file is divided into a header that contain the maximum size that a string can take , a float ( that i ignor), a len and the bytes of the string , what i do its going thorugh the binary file reading the len and then reading the strign and writing direct into the string buffer , rememeber that string are jsust a list of character connected by points .
