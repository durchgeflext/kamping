#pragma once

#include "kamping/communicator.hpp"

template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename... Args>
void kamping::Communicator<DefaultContainerType, Plugins...>::sendrecv(Args... args) {

}