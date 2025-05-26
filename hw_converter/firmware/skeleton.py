from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.design.ports import Port


class Skeleton:
    def __init__(
        self,
        network_name: str,
        x_addr_width: int,
        y_addr_width: int,
        x_data_width: int,
        y_data_width: int,
        x_count: int,
        y_count: int,
    ):
        self.name = "skeleton"
        self._network_name = network_name

        self._x_data_width = str(x_data_width)
        self._y_data_width = str(y_data_width)
        self._x_addr_width = str(x_addr_width)
        self._y_addr_width = str(y_addr_width)
        self._x_count = str(x_count)
        self._y_count = str(y_count)

    def save_to(self, destination: Path):
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="skeleton.tpl.vhd",
            parameters=dict(
                network_name=self._network_name,
                x_data_width=self._x_data_width,
                y_data_width=self._y_data_width,
                x_addr_width=self._x_addr_width,
                y_addr_width=self._y_addr_width,
                x_count=self._x_count,
                y_count=self._y_count,
            ),
        )
        file = destination.create_subpath("skeleton").as_file(".vhd")
        file.write(template)
