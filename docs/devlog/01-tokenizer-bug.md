# How to read a tokenizer?

I needed to read 32,000 tokens out of a binary file. Each one is a lenght followed by that many bytes. So I wrote ```read(reinterpret_cast<char*>(&s), len)``` and got a bus error with a stack trace poiting at a line that wasn't read.
```&s``` isn't the characters. It's the address of the ```std::string``` object, a small struct holding a pointer, a size and a capacity. I was writing bytes files over those internals. The string's pointer became garbage, and the crash only happen later when something derefenced it.
AddressSanitazer couldn't help. I'd written the correct number of bytes into valid allocation. It just went to the semantically wrong place.
The fix is ```&s[0]``` , the character buffer.
