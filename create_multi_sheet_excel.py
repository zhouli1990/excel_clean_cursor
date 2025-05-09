# 检查是否存在并读取create_multi_sheet_excel.py文件
# 如果不存在，创建该文件并添加必要的函数
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_multi_sheet_excel(
    raw_data_df: pd.DataFrame,
    output_file_path: str,
    sheet_configs: dict = None,
    source_column_name: str = "来源",
) -> str:
    """
    将DataFrame数据保存为多sheet的Excel文件，按照sheet_configs的配置分类。

    Args:
        raw_data_df (pd.DataFrame): 包含原始数据的DataFrame
        output_file_path (str): 输出Excel文件的路径
        sheet_configs (dict): 各sheet的配置信息
        source_column_name (str): 来源列的名称，默认为"来源"

    Returns:
        str: 输出文件的路径
    """
    try:
        if raw_data_df.empty:
            logger.warning("输入的DataFrame为空，无法创建Excel文件")
            return None

        # 记录输入数据的信息
        logger.info(f"创建多Sheet Excel文件，输入数据有 {len(raw_data_df)} 行")
        logger.info(f"输入数据列: {list(raw_data_df.columns)}")

        # 检查来源列是否存在
        if source_column_name in raw_data_df.columns:
            logger.info(f"来源列 '{source_column_name}' 存在于输入数据中")
            # 记录来源列的唯一值
            source_values = raw_data_df[source_column_name].unique()
            logger.info(f"来源列的唯一值: {source_values}")
        else:
            logger.warning(f"来源列 '{source_column_name}' 不存在于输入数据中")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # 创建Excel写入器
        with pd.ExcelWriter(output_file_path, engine="openpyxl") as writer:
            # 原始数据写入"原始数据"表
            raw_data_df.to_excel(writer, sheet_name="原始数据", index=False)
            logger.info(f"已写入原始数据，共 {len(raw_data_df)} 行")

            # 如果提供了sheet配置，按照配置分表
            if sheet_configs and isinstance(sheet_configs, dict):
                for sheet_name, config in sheet_configs.items():
                    if "filter" in config and callable(config["filter"]):
                        filtered_df = config["filter"](raw_data_df)
                        filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        logger.info(f"已写入 {sheet_name} 表，共 {len(filtered_df)} 行")

            # 按来源分表
            if source_column_name in raw_data_df.columns:
                # 获取来源列的唯一值
                sources = raw_data_df[source_column_name].unique()

                # 为每个来源创建单独的sheet
                for source in sources:
                    if not source or pd.isna(source):
                        continue

                    # 过滤出当前来源的数据
                    source_df = raw_data_df[raw_data_df[source_column_name] == source]

                    if not source_df.empty:
                        # 创建有效的sheet名称（Excel限制sheet名长度为31个字符）
                        # 如果是文件名，取基础名（不含扩展名）
                        if isinstance(source, str) and "." in source:
                            base_name = os.path.splitext(source)[0]
                            sheet_name = base_name[:30]  # 保留前30个字符
                        else:
                            sheet_name = str(source)[:30]  # 转为字符串并保留前30个字符

                        # 将数据写入对应的sheet
                        source_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        logger.info(
                            f"已写入来源表 '{sheet_name}'，共 {len(source_df)} 行"
                        )

        logger.info(f"多Sheet Excel文件创建成功: {output_file_path}")
        return output_file_path

    except Exception as e:
        logger.error(f"创建多Sheet Excel文件时出错: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise
