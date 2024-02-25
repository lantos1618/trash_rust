#include <stdint.h>

int main() {
    int64_t x = 0;
    while (x < 420000000)
    {
        x = x + 1;
    }
    
    return x;
}