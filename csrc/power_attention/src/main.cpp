#include "index.h"
#include "state.h"


int main() {
    auto m = power_attention::MultiIndex<4, 8>();
    m.print_index();
}