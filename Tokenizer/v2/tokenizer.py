import ctypes
import os
import subprocess
import platform
import struct

class Tokenizer:
    def __init__(self, bin_path="id_to_token.bin", merges_path="merges.bin"):
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.lib_path = self._compile_if_needed()
        self.lib = ctypes.CDLL(self.lib_path)
        
        # Define C API signatures
        self.lib.create_tokenizer.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.create_tokenizer.restype = ctypes.c_void_p
        
        self.lib.encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.encode.restype = ctypes.POINTER(ctypes.c_int)
        
        self.lib.free_tokens.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.free_tokens.restype = None
        
        self.lib.free_tokenizer.argtypes = [ctypes.c_void_p]
        self.lib.free_tokenizer.restype = None
        
        # Load C++ tokenizer object
        bin_c = os.path.join(self.dir_path, bin_path).encode('utf-8')
        merges_c = os.path.join(self.dir_path, merges_path).encode('utf-8')
        self.obj = self.lib.create_tokenizer(bin_c, merges_c)
        
        # Pre-cache binary decoding map natively in Python for instant decoding
        # (decoding is trivial, no need to offload to C++)
        self.bin_cache = {}
        with open(os.path.join(self.dir_path, bin_path), "rb") as f:
            total = struct.unpack("I", f.read(4))[0]
            for i in range(total):
                length = struct.unpack("I", f.read(4))[0]
                self.bin_cache[i] = f.read(length)
        
    def _compile_if_needed(self):
        ext = ".dylib" if platform.system() == "Darwin" else ".so"
        lib_name = f"inference{ext}"
        lib_path = os.path.join(self.dir_path, lib_name)
        cpp_path = os.path.join(self.dir_path, "inference.cpp")
        
        if not os.path.exists(lib_path):
            print(f"Compiling C++ backend to {lib_name}...")
            cmd = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", cpp_path, "-o", lib_path]
            subprocess.run(cmd, check=True)
            
        return lib_path
        
    def encode(self, text: str) -> list[int]:
        out_len = ctypes.c_int(0)
        # Convert string to utf-8 bytes so C++ regex processes raw bytes perfectly
        text_bytes = text.encode('utf-8')
        res_ptr = self.lib.encode(self.obj, text_bytes, ctypes.byref(out_len))
        
        if not res_ptr:
            return []
            
        count = out_len.value
        tokens = [res_ptr[i] for i in range(count)]
        
        # Free memory allocated by C++ new[]
        self.lib.free_tokens(res_ptr)
        return tokens
        
    def decode(self, tokens: list[int]) -> str:
        b = b""
        for t in tokens:
            if t in self.bin_cache:
                b += self.bin_cache[t]
        return b.decode("utf-8", errors="replace")

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            self.lib.free_tokenizer(self.obj)

if __name__ == "__main__":
    # Test block
    print("Initializing Tokenizer (will compile C++ backend on first run)...")
    tok = Tokenizer()
    
    text = "Hello world! This is a incredibly fast BPE test powered by C++."
    print("\nOriginal Text:")
    print(text)
    
    encoded = tok.encode(text)
    print("\nEncoded IDs:")
    print(encoded)
    
    decoded = tok.decode(encoded)
    print("\nDecoded Text:")
    print(decoded)
    
    assert text == decoded, "Decoded text does not match original!"
    print("\nSUCCESS! Tokenizer is working perfectly.")
