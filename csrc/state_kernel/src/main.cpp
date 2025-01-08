#include "index.h"
#include "state.h"


int main() {
    auto m = state_kernel::MultiIndex<4, 8>();
    m.print_index();
}