from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.vhdl.design.design import Design
from hw_converter.firmware.skeleton import Skeleton
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)


class GetFirmware:
    def __init__(
        self,
        hw_version: str,
        network_design: Design,
    ) -> None:
        self.hw_version = hw_version
        self._skeleton = Skeleton(
            network_name=network_design.name,
            x_count=network_design._x_count,
            y_count=network_design._y_count,
            x_addr_width=network_design._x_addr_width,
            y_addr_width=network_design._y_addr_width,
            x_data_width=network_design._data_width,
            y_data_width=network_design._data_width,
        )
        self._x_count = network_design._x_count

    def _save_middleware_files(self, destination: Path) -> None:

        middleware_files = [
            "icapInterface",
            "InterfaceStateMachine",
            "middleware",
            "spi_slave",
            "UserLogicInterface",
        ]
        middleware_files = [f"{self.hw_version}_{name}" for name in middleware_files]

        for name in middleware_files:
            template = InProjectTemplate(
                package=module_to_package(self.__module__),
                file_name=name + ".vhd",
                parameters={},
            )
            file = destination.create_subpath(name).as_file(".vhd")
            file.write(template)

    def _save_top_files(self, destination: Path) -> None:
        file_names = [f"{self.hw_version}_top_reconfig.vhd"]
        if self.hw_version == "env5se":
            file_names.append(f"{self.hw_version}_top_reconfig_tb.vhd")

        for file_name in file_names:
            if file_name.endswith("_tb.vhd"):
                parameters = {"x_count": str(self._x_count)}
            else:
                parameters = {}

            template = InProjectTemplate(
                package=module_to_package(self.__module__),
                file_name=file_name,
                parameters=parameters,
            )
            file = destination.create_subpath(file_name).as_file("")
            file.write(template)

    def _save_amd_constraints(self, destination: Path) -> None:
        file_names = ["env5_resource.xdc", "env5_power.xdc"]
        for file_name in file_names:
            template = InProjectTemplate(
                package=module_to_package(self.__module__),
                file_name=file_name,
                parameters={},
            )
            file = destination.create_subpath(file_name).as_file("")
            file.write(template)

    def _save_lattice_constraints(self, destination: Path) -> None:
        file_names = ["env5se_clock.sdc", "env5se_pins.pdc"]
        for file_name in file_names:
            template = InProjectTemplate(
                package=module_to_package(self.__module__),
                file_name=file_name,
                parameters={},
            )
            file = destination.create_subpath(file_name).as_file("")
            file.write(template)

    def _save_srcs(self, destination: Path) -> None:
        self._skeleton.save_to(destination)
        self._save_middleware_files(destination)
        self._save_top_files(destination)

    def _save_constraints(self, destination: Path) -> None:

        if self.hw_version == "env5":
            self._save_amd_constraints(destination)
        elif self.hw_version == "env5se":
            self._save_lattice_constraints(destination)
        else:
            raise ValueError(f"Invalid hw_version: {self.hw_version}")

    def save_to(self, destination_dir: Path) -> None:
        self._save_constraints(
            OnDiskPath(name=f"constraints", parent=str(destination_dir))
        )
        self._save_srcs(OnDiskPath(name=f"firmware", parent=str(destination_dir)))
