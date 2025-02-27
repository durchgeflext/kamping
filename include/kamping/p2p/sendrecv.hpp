#pragma once

#include "helpers.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameter_types.hpp"

template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename recv_value_type_tparam, typename... Args>
void kamping::Communicator<DefaultContainerType, Plugins...>::sendrecv(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, destination),
        KAMPING_OPTIONAL_PARAMETERS(send_count, send_tag, send_mode, send_type, recv_buf, source, recv_count,
                            recv_tag, recv_type, ParameterType::status)
    );
    auto send_buf = internal::select_parameter_type<internal::ParameterType::send_buf>(args...)
                    .template construct_buffer_or_rebind<UnusedRebindContainer, serialization_support_tag>();
  constexpr bool is_serialization_used = internal::buffer_uses_serialization<decltype(send_buf)>;
  if constexpr (is_serialization_used) {
    KAMPING_UNSUPPORTED_PARAMETER(Args, send_count, when using serialization);
    KAMPING_UNSUPPORTED_PARAMETER(Args, send_type, when using serialization);
    send_buf.underlying().serialize();
  }
  using send_value_type = typename std::remove_reference_t<decltype(send_buf)>::value_type;

  auto send_type = internal::determine_mpi_send_datatype<send_value_type>(args...);

  using default_send_count_type = decltype(kamping::send_count_out());
  auto send_count =
      internal::select_parameter_type_or_default<internal::ParameterType::send_count, default_send_count_type>(
          {},
          args...
      )
          .construct_buffer_or_rebind();
  if constexpr (has_to_be_computed<decltype(send_count)>) {
    send_count.underlying() = asserting_cast<int>(send_buf.size());
  }

    auto const&    destination = internal::select_parameter_type<internal::ParameterType::destination>(args...);
    constexpr auto rank_type   = std::remove_reference_t<decltype(destination)>::rank_type;
    static_assert(
        rank_type == RankType::value || rank_type == RankType::null,
        "Please provide an explicit destination or destination(ranks::null)."
    );

    using default_tag_buf_type = decltype(kamping::tag(this->default_tag()));

    auto&& tag_param = internal::select_parameter_type_or_default<internal::ParameterType::tag, default_tag_buf_type>(
        std::tuple(this->default_tag()),
        args...
    );

    // this ensures that the user does not try to pass MPI_ANY_TAG, which is not allowed for sends
    static_assert(
        std::remove_reference_t<decltype(tag_param)>::tag_type == TagType::value,
        "Please provide a tag for the message."
    );
    int tag = tag_param.tag();
    KASSERT(
        Environment<>::is_valid_tag(tag),
        "invalid tag " << tag << ", must be in range [0, " << Environment<>::tag_upper_bound() << "]"
    );

    using send_mode_obj_type = decltype(internal::select_parameter_type_or_default<
                                        internal::ParameterType::send_mode,
                                        internal::SendModeParameter<internal::standard_mode_t>>(std::tuple(), args...));
    using send_mode          = typename std::remove_reference_t<send_mode_obj_type>::send_mode;

    // RankType::null is valid, RankType::any is not.
    KASSERT(is_valid_rank_in_comm(destination, *this, true, false), "Invalid destination rank.");

    [[maybe_unused]] int err = MPI_Sendrecv(
    send_buf.data(),                 // send_buf
    send_count.get_single_element(), // send_count
    send_type.get_single_element(),  // send_type
    destination.rank_signed(),       // destination
    tag,                             // tag
    this->mpi_communicator()
);
    this->mpi_error_hook(err, "MPI_Send");
}