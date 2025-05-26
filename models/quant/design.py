from itertools import chain

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port
from elasticai.creator.vhdl.shared_designs.rom import Rom


class Transformer(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        input_linear: object,
        inputs_pos_info: list[list[int]],
        pos_info_addition: object,
        encoder: object,
        avg_pooling: object,
        output_linear: object,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._input_linear = input_linear
        self._inputs_pos_info = inputs_pos_info
        self._pos_info_addition = pos_info_addition
        self._encoder = encoder
        self._avg_pooling = avg_pooling
        self._output_linear = output_linear

        self._work_library_name = work_library_name

        self.input_linear_design = self._input_linear.create_design(
            name=self._input_linear.name
        )
        self.pos_info_addition_design = self._pos_info_addition.create_design(
            name=self._pos_info_addition.name
        )
        self.encoder_design = self._encoder.sequential.create_design(
            name=self._encoder.name
        )
        self.avg_pooling_design = self._avg_pooling.create_design(
            name=self._avg_pooling.name
        )
        self.output_linear_design = self._output_linear.create_design(
            name=self._output_linear.name
        )
        self._x_count = self.input_linear_design._x_count
        self._y_count = self.output_linear_design._y_count

        self._x_addr_width = self.input_linear_design._x_addr_width
        self._y_addr_width = self.output_linear_design._y_addr_width

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self._x_count,
            y_count=self._y_count,
        )

    def save_to(self, destination: Path) -> None:

        self.input_linear_design.save_to(
            destination.create_subpath(self.input_linear_design.name)
        )

        self.pos_info_addition_design.save_to(
            destination.create_subpath(self.pos_info_addition_design.name)
        )
        self.encoder_design.save_to(
            destination.create_subpath(self.encoder_design.name)
        )
        self.avg_pooling_design.save_to(
            destination.create_subpath(self.avg_pooling_design.name)
        )
        self.output_linear_design.save_to(
            destination.create_subpath(self.output_linear_design.name)
        )

        destination = destination.create_subpath(self.name)
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="transformer.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self.input_linear_design._x_addr_width),
                y_addr_width=str(self.output_linear_design._y_addr_width),
                input_linear_x_addr_width=str(self.input_linear_design._x_addr_width),
                input_linear_y_addr_width=str(self.input_linear_design._y_addr_width),
                add_pos_info_x_addr_width=str(
                    self.pos_info_addition_design._x_addr_width
                ),
                add_pos_info_y_addr_width=str(
                    self.pos_info_addition_design._y_addr_width
                ),
                gap_x_addr_width=str(self.avg_pooling_design._x_addr_width),
                gap_y_addr_width=str(self.avg_pooling_design._y_addr_width),
                output_linear_x_addr_width=str(self.output_linear_design._x_addr_width),
                output_linear_y_addr_width=str(self.output_linear_design._y_addr_width),
                input_linear_name=self.input_linear_design.name,
                add_pos_info_name=self.pos_info_addition_design.name,
                pos_info_rom_name="pos_info_rom",
                encoder_name=self.encoder_design.name,
                avgpooling_name=self.avg_pooling_design.name,
                output_linear_name=self.output_linear_design.name,
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        rom_pos_info = Rom(
            name="pos_info_rom",
            data_width=self._data_width,
            values_as_integers=_flatten_params(self._inputs_pos_info),
        )
        rom_pos_info.save_to(destination.create_subpath("pos_info_rom"))

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="transformer_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self.input_linear_design._x_addr_width),
                y_addr_width=str(self.output_linear_design._y_addr_width),
                x_count=str(self.input_linear_design._x_count),
                y_count=str(self.output_linear_design._y_count),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)

        # template_skeleton = InProjectTemplate(
        #     package=module_to_package(self.__module__),
        #     file_name="skeleton.tpl.vhd",
        #     parameters=dict(
        #         network_name=self.name,
        #         data_width=str(self._data_width),
        #         x_addr_width=str(self.input_linear_design._x_addr_width),
        #         x_count=str(self.input_linear_design._x_count),
        #         y_addr_width=str(self.output_linear_design._y_addr_width),
        #         work_library_name=self._work_library_name,
        #     ),
        # )
        # destination.create_subpath("skeleton").as_file(".vhd").write(template_skeleton)


def _flatten_params(params: list[list[int]]) -> list[int]:
    return list(chain(*params))
