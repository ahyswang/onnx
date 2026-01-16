import onnx

# list_onnx_ops. py

import onnx
from onnx import defs
import argparse
from collections import defaultdict


def list_all_ops():
    """列出所有算子"""
    all_schemas = defs.get_all_schemas()
    
    # 按域分组
    by_domain = defaultdict(list)
    for schema in all_schemas:
        by_domain[schema.domain].append(schema)
    
    print(f"\n{'=' * 80}")
    print(f"ONNX 支持的算子总数: {len(all_schemas)}")
    print(f"{'=' * 80}\n")
    
    for domain in sorted(by_domain.keys()):
        domain_name = domain if domain else "ONNX Standard"
        schemas = by_domain[domain]
        
        print(f"\n域:  {domain_name} ({len(schemas)} 个算子)")
        print("-" * 80)
        
        for schema in sorted(schemas, key=lambda x: x.name):
            print(f"  {schema.name:30} (v{schema.since_version})")


def list_ops_by_domain(domain=""):
    """按域列出算子"""
    all_schemas = defs.get_all_schemas()
    domain_schemas = [s for s in all_schemas if s.domain == domain]
    
    domain_name = domain if domain else "ONNX Standard"
    print(f"\n{'=' * 80}")
    print(f"域:  {domain_name} ({len(domain_schemas)} 个算子)")
    print(f"{'=' * 80}\n")
    
    print(f"{'算子名':<30} {'版本':<8} {'输入':<10} {'输出':<10}")
    print("-" * 80)
    
    for schema in sorted(domain_schemas, key=lambda x: x.name):
        inputs = f"{schema.min_input}-{schema.max_input if schema.max_input < 1000 else 'N'}"
        outputs = f"{schema.min_output}-{schema.max_output if schema.max_output < 1000 else 'N'}"
        print(f"{schema.name:<30} {schema.since_version:<8} {inputs:<10} {outputs:<10}")


def show_op_detail(op_name, domain=""):
    """显示算子详细信息"""
    try: 
        schema = defs.get_schema(op_name, domain=domain)
        
        print(f"\n{'=' * 80}")
        print(f"算子:  {schema.name}")
        print(f"{'=' * 80}")
        print(f"域: {schema. domain if schema.domain else 'ONNX Standard'}")
        print(f"引入版本: {schema.since_version}")
        print(f"支持级别: {schema.support_level}")
        
        print(f"\n文档:")
        print("-" * 80)
        print(schema.doc if schema.doc else "无文档")
        
        print(f"\n输入 ({schema.min_input} 到 {schema.max_input if schema.max_input < 1000 else '不限'}):")
        print("-" * 80)
        
        if schema.inputs:
            for i, inp in enumerate(schema.inputs):
                option = inp.option if hasattr(inp, 'option') else "必需"
                print(f"  [{i}] {inp.name} ({option})")
                print(f"      类型: {inp.type_str}")
                print(f"      描述: {inp.description}")
        else:
            print("  无输入")
        
        print(f"\n输出 ({schema.min_output} 到 {schema.max_output if schema.max_output < 1000 else '不限'}):")
        print("-" * 80)
        if schema.outputs:
            for i, out in enumerate(schema.outputs):
                print(f"  [{i}] {out.name}")
                print(f"      类型: {out.type_str}")
                print(f"      描述: {out.description}")
        else:
            print("  无输出")
        
        print(f"\n属性:")
        print("-" * 80)
        if schema.attributes:
            for attr_name, attr in schema.attributes. items():
                required = "必需" if attr.required else "可选"
                default = f" (默认: {attr.default_value})" if hasattr(attr, 'default_value') and attr.default_value else ""
                print(f"  - {attr_name} ({required}){default}")
                print(f"    类型: {attr.type}")
                print(f"    描述: {attr.description}")
        else:
            print("  无属性")
        
        print(f"\n类型约束:")
        print("-" * 80)
        if schema.type_constraints:
            for tc in schema.type_constraints:
                print(f"  - {tc.type_param_str}:  {', '.join(tc.allowed_type_strs)}")
        else:
            print("  无类型约束")
        
        print(f"{'=' * 80}\n")
        
    except Exception as e:
        print(f"错误: {e}")


def search_ops(keyword):
    """搜索包含关键字的算子"""
    all_schemas = defs.get_all_schemas()
    keyword_lower = keyword.lower()
    
    matched = [s for s in all_schemas if keyword_lower in s.name. lower()]
    
    print(f"\n搜索 '{keyword}' 的结果 ({len(matched)} 个):")
    print("=" * 80)
    
    for schema in sorted(matched, key=lambda x: x.name):
        domain_name = schema.domain if schema.domain else "ONNX Standard"
        print(f"{schema.name:<30} | 域: {domain_name: <20} | v{schema.since_version}")


def list_ops_by_version(version):
    """列出特定版本支持的算子"""
    all_schemas = defs.get_all_schemas()
    version_schemas = [s for s in all_schemas 
                       if s.since_version <= version and s.domain == ""]
    
    print(f"\nONNX Opset {version} 支持的算子 ({len(version_schemas)} 个):")
    print("=" * 80)
    
    # 按引入版本分组
    by_version = defaultdict(list)
    for schema in version_schemas:
        by_version[schema.since_version].append(schema.name)
    
    for v in sorted(by_version. keys()):
        ops = sorted(by_version[v])
        print(f"\n版本 {v} ({len(ops)} 个):")
        for i in range(0, len(ops), 5):
            print("  " + ", ".join(f"{op:<20}" for op in ops[i: i+5]))


def check_op_exists(op_name, domain=""):
    """检查算子是否存在"""
    try:
        schema = defs. get_schema(op_name, domain=domain)
        domain_name = domain if domain else "ONNX Standard"
        print(f"✓ 算子 '{op_name}' 存在于域 '{domain_name}' (版本 {schema.since_version})")
        return True
    except Exception: 
        domain_name = domain if domain else "ONNX Standard"
        print(f"✗ 算子 '{op_name}' 不存在于域 '{domain_name}'")
        return False
    
#python ./onnx_infershape/list_onnx_ops.py --domain com.nebula
#python ./onnx_infershape/list_onnx_ops.py --check QLinearAdd --op-domain com.nebula
#python ./onnx_infershape/list_onnx_ops.py --detail  QLinearAdd --op-domain com.nebula

def main():
    parser = argparse.ArgumentParser(description="ONNX 算子查询工具")
    parser.add_argument("--list", action="store_true", help="列出所有算子")
    parser.add_argument("--domain", type=str, default=None, help="按域列出算子")
    parser.add_argument("--detail", type=str, help="显示算子详细信息")
    parser.add_argument("--search", type=str, help="搜索算子")
    parser.add_argument("--version", type=int, help="列出特定 opset 版本的算子")
    parser.add_argument("--check", type=str, help="检查算子是否存在")
    parser.add_argument("--op-domain", type=str, default="", help="算子所属域（与 --detail 或 --check 配合使用）")
    
    args = parser.parse_args()

    if args.list:
        list_all_ops()
    elif args.domain is not None:
        list_ops_by_domain(args.domain)
    elif args.detail:
        show_op_detail(args.detail, args.op_domain)
    elif args.search:
        search_ops(args.search)
    elif args.version:
        list_ops_by_version(args.version)
    elif args.check:
        check_op_exists(args.check, args.op_domain)
    else:
        # 默认：列出所有算子
        list_all_ops()


if __name__ == "__main__": 
    main()