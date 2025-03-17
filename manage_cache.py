import asyncio
from typing import List, Dict
import click
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich.style import Style
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align
from datetime import datetime
from pymilvus import connections, utility
from .vector_store import VectorStore
import math
import time
from loguru import logger
from .milvus_client import MilvusClientSingleton
import json

console = Console()

def create_header() -> Panel:
    """헤더 생성"""
    title = """
    ██╗      ██████╗  ██████╗  ██████╗ ███████╗ █████╗ ██╗
    ██║     ██╔═══██╗██╔════╝ ██╔═══██╗██╔════╝██╔══██╗██║
    ██║     ██║   ██║██║  ███╗██║   ██║███████╗███████║██║
    ██║     ██║   ██║██║   ██║██║   ██║╚════██║██╔══██║██║
    ███████╗╚██████╔╝╚██████╔╝╚██████╔╝███████║██║  ██║██║
    ╚══════╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝
    """
    
    subtitle = """
    ╔═══════════════════════════════════════════════╗
     ║         Cache Management System v1.0          ║
    ╚═══════════════════════════════════════════════╝
    """
    
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_row(Text(title, style="deep_sky_blue1", justify="center"))
    grid.add_row(Text(subtitle, style="steel_blue1", justify="center"))
    
    return Panel(
        grid,
        style="white",
        border_style="steel_blue3",
        padding=(1, 1)
    )

def create_footer() -> Panel:
    """푸터 생성"""
    footer_art = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  [?] Help  [Q] Quit  [←] Back  [→] Next  [↵] Select  [⌫] Cancel   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    
    grid = Table.grid(expand=True)
    grid.add_column(justify="center")
    grid.add_row(Text(footer_art, style="steel_blue1"))
    grid.add_row(Text("LogosAI © 2024 - Powered by Milvus", style="grey70"))
    
    return Panel(grid, border_style="steel_blue3", style="white")

def create_menu() -> Panel:
    """메뉴 패널 생성"""
    menu_art = """
    ┌─────────────────────────────────────────┐
    │             Available Commands           │
    └─────────────────────────────────────────┘
    """
    
    menu_items = [
        ("1", "📋 List Cache Entries", "View all cache entries with pagination"),
        ("2", "🔍 Search Entries", "Search cache entries by query or result"),
        ("3", "🗑️  Delete Entry", "Remove a specific cache entry"),
        ("4", "📊 Statistics", "View cache statistics"),
        ("5", "🔄 Reset Collection", "Reset entire cache collection"),
        ("6", "💣 Drop Collection", "Permanently delete collection"),
        ("7", "🚪 Exit", "Exit the application")
    ]
    
    menu_table = Table(show_header=False, expand=True, box=None)
    menu_table.add_column("Key", style="sky_blue2", width=4)
    menu_table.add_column("Option", style="steel_blue1")
    menu_table.add_column("Description", style="grey70")
    
    for key, option, desc in menu_items:
        menu_table.add_row(f"[{key}]", option, desc)
    
    return Panel(
        Align.center(Group(
            Text(menu_art, style="steel_blue1"),
            menu_table
        )),
        title="[bold steel_blue1]Menu Options",
        border_style="steel_blue3",
        padding=(1, 2)
    )

class CacheManager:
    def __init__(self):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Connecting to Milvus...", total=None)
            self.milvus_client = MilvusClientSingleton.get_instance()
            self.collection = self.milvus_client.get_collection()
            self.collection_name = self.milvus_client.collection_name
            self.page_size = 20

    async def get_cache_entries(self, page: int = 1) -> tuple[List[Dict], int]:
        """캐시 엔트리 조회 (페이지네이션)"""
        try:
            total_count = self.collection.num_entities
            logger.info(f"Total entries in collection: {total_count}")
            
            if total_count == 0:
                logger.warning("No entries found in collection")
                return [], 0
            
            offset = (page - 1) * self.page_size
            logger.info(f"Querying with offset={offset}, limit={self.page_size}")
            
            # 전체 데이터 조회 시도
            results = self.collection.query(
                expr="",
                output_fields=["id", "query_text", "result_text", "email", 
                             "created_at", "project_id", "metadata"],
                limit=100
            )
            logger.info(f"Retrieved {len(results)} entries from collection")
            
            # 결과 로깅 추가
            logger.debug(f"First result metadata: {results[0].get('metadata') if results else 'No results'}")
            
            # 페이지네이션 적용
            start_idx = offset
            end_idx = min(offset + self.page_size, len(results))
            page_entries = results[start_idx:end_idx]
            
            processed_results = []
            for r in page_entries:
                metadata = r.get("metadata", {})
                # metadata가 문자열인 경우 파싱
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse metadata JSON: {metadata}")
                        metadata = {}
                
                processed_results.append({
                    "id": r["id"],
                    "query_text": r["query_text"],
                    "result_text": r["result_text"],
                    "email": r["email"],
                    "created_at": r["created_at"],
                    "project_id": r["project_id"],
                    "metadata": metadata  # 파싱된 메타데이터 포함
                })
            
            return processed_results, total_count
            
        except Exception as e:
            logger.error(f"Failed to get cache entries: {e}")
            return [], 0

    async def delete_entry(self, entry_id: str) -> bool:
        """캐시 엔트리 삭제"""
        try:
            collection = self.collection
            collection.load()
            
            expr = f'id == "{entry_id}"'
            collection.delete(expr)
            return True

        except Exception as e:
            console.print(f"[red]Error deleting cache entry: {e}[/red]")
            return False

    async def search_entries(self, query: str) -> List[Dict]:
        """캐시 엔트리 검색"""
        try:
            collection = self.collection
            collection.load()
            
            expr = f'query_text like "%{query}%" or result_text like "%{query}%"'
            results = collection.query(
                expr=expr,
                output_fields=["id", "query_text", "email", "created_at", "project_id", "metadata", "result_text"],
                limit=self.page_size,
                sort_fields=["created_at"],  # 정렬 필드
                sort_orders=["DESC"]         # 내림차순
            )

            entries = []
            for r in results:
                entries.append({
                    "id": r["id"],
                    "query_text": r["query_text"][:100] + "..." if len(r["query_text"]) > 100 else r["query_text"],
                    "email": r["email"],
                    "created_at": r["created_at"],
                    "project_id": r["project_id"],
                    "hit_count": r["metadata"].get("hit_count", 0),
                    "result_text": r["result_text"][:100] + "..." if len(r["result_text"]) > 100 else r["result_text"]
                })

            return entries

        except Exception as e:
            console.print(f"[red]Error searching cache entries: {e}[/red]")
            return []

def display_entries(entries: List[Dict], current_page: int, total_pages: int) -> Panel:
    """캐시 엔트리 목록 표시"""
    if not entries:
        return Panel("[yellow]No entries found[/yellow]", title="Cache Entries")
    
    table = Table(
        show_header=True,
        header_style="bold sky_blue2",
        border_style="steel_blue3",
        width=console.width - 4
    )
    
    table.add_column("No", style="grey70", width=4)
    table.add_column("ID", style="grey70", width=10)
    table.add_column("Query", style="steel_blue1", width=30)
    table.add_column("Result", style="steel_blue1", width=40)
    table.add_column("Email", style="grey70", width=20)
    table.add_column("Created At", style="grey70", width=20)
    table.add_column("Hits", style="sky_blue2", justify="right", width=5)
    
    for idx, entry in enumerate(entries, 1):
        metadata = entry.get("metadata", {})
        hit_count = metadata.get("hit_count", 0) if isinstance(metadata, dict) else 0
        
        table.add_row(
            str(idx),
            truncate_text(entry.get("id", ""), 10),
            truncate_text(entry.get("query_text", ""), 30),
            truncate_text(entry.get("result_text", ""), 40),
            truncate_text(entry.get("email", ""), 30),
            entry.get("created_at", ""),
            str(hit_count)
        )
    
    return Panel(
        Group(
            table,
            Align.right(Text(f"Page {current_page} of {total_pages}", style="italic"))
        ),
        title="Cache Entries",
        border_style="steel_blue3"
    )

def display_entry_detail(entry: Dict) -> Panel:
    """캐시 엔트리 상세 정보 표시"""
    try:
        # metadata 처리
        metadata = entry.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse metadata JSON: {metadata}")
                metadata = {}
        
        # logger.debug(f"Processing metadata: {metadata}")
        
        detail_table = Table(show_header=False, border_style="steel_blue3", width=console.width - 4)
        detail_table.add_column("Field", style="sky_blue2", width=20)
        detail_table.add_column("Value", style="steel_blue1")
        
        # 기본 정보 표시
        detail_table.add_row("ID", str(entry.get("id", "")))
        detail_table.add_row("Query", str(entry.get("query_text", "")))
        detail_table.add_row("Result", str(entry.get("result_text", "")))
        detail_table.add_row("Email", str(entry.get("email", "")))
        detail_table.add_row("Created At", str(entry.get("created_at", "")))
        detail_table.add_row("Hit Count", str(metadata.get("hit_count", 0)))
        
        # Cache Management 정보 표시
        cache_mgmt = metadata.get("cache_management", {})
        if isinstance(cache_mgmt, str):
            try:
                cache_mgmt = json.loads(cache_mgmt)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse cache_management JSON: {cache_mgmt}")
                cache_mgmt = {}
                
        # logger.debug(f"Processing cache_management: {cache_mgmt}")
        
        if cache_mgmt:
            cache_table = Table(show_header=True, border_style="steel_blue3")
            cache_table.add_column("Property", style="sky_blue2")
            cache_table.add_column("Value", style="steel_blue1")
            
            # 기본 캐시 정보
            cache_table.add_row("Should Cache", "✅ Yes" if cache_mgmt.get("should_cache") else "❌ No")
            cache_table.add_row("Category", str(cache_mgmt.get("category", "N/A")))
            cache_table.add_row("Priority", str(cache_mgmt.get("cache_priority", "N/A")))
            cache_table.add_row("Expiration", str(cache_mgmt.get("expiration", "N/A")))
            cache_table.add_row("Reasoning", Text(str(cache_mgmt.get("reasoning", "N/A")), style="italic"))
            
            # 캐시 메타데이터 정보
            cache_metadata = cache_mgmt.get("metadata", {})
            if cache_metadata:
                cache_table.add_row("Reuse Potential", str(cache_metadata.get("reuse_potential", "N/A")))
                cache_table.add_row("Accuracy Level", str(cache_metadata.get("accuracy_level", "N/A")))
                cache_table.add_row("Update Frequency", str(cache_metadata.get("update_frequency", "N/A")))
            
            detail_table.add_row("Cache Management", cache_table)
        
        # 참조 정보 표시
        if metadata.get("references"):
            refs_table = Table(show_header=True, border_style="steel_blue3")
            refs_table.add_column("Ref. File Name, Page Number", width=120)
            # refs_table.add_column("File", width=30)
            # refs_table.add_column("Page", width=10)
            
            for ref in metadata["references"]:
                if isinstance(ref, dict):
                    refs_table.add_row(
                        str(ref.get("text", "")),
                        # str(ref.get("file_name", "")),
                        # str(ref.get("page", ""))
                    )
        
            detail_table.add_row("References", refs_table)
        
        # 작업 옵션 추가
        actions_table = Table.grid(padding=1)
        actions_table.add_row("[d] Delete Entry    [b] Back")
        
        return Panel(
            Group(
                detail_table,
                Text(""),  # 간격 추가
                Align.center(actions_table)
            ),
            title="[bold sky_blue2]Entry Detail",
            border_style="steel_blue3"
        )
        
    except Exception as e:
        logger.error(f"Error in display_entry_detail: {e}")
        return Panel(f"[red]Error displaying entry details: {e}[/red]")

def truncate_text(text: str, max_length: int) -> str:
    """텍스트를 지정된 길이로 자르고 말줄임표 추가"""
    if not text:
        return ""
    return (text[:max_length - 3] + "...") if len(text) > max_length else text

async def display_statistics(cache_manager: CacheManager):
    """캐시 통계 표시"""
    try:
        collection = cache_manager.collection
        total_count = collection.num_entities
        
        # 기본 통계 테이블
        stats = Table(show_header=False, border_style="bright_blue")
        stats.add_column("Metric", style="yellow")
        stats.add_column("Value", style="white")
        
        stats.add_row("Total Entries", str(total_count))
        stats.add_row("Collection Name", cache_manager.collection_name)
        stats.add_row("Vector Dimension", str(cache_manager.milvus_client.dim))
        
        # 프로젝트별 통계
        project_stats = collection.query(
            expr="",
            output_fields=["project_id", "email"],
            limit=10000
        )
        
        if project_stats:
            project_count = {}
            user_count = set()
            for entry in project_stats:
                project_count[entry["project_id"]] = project_count.get(entry["project_id"], 0) + 1
                user_count.add(entry["email"])
            
            stats.add_row("Total Projects", str(len(project_count)))
            stats.add_row("Total Users", str(len(user_count)))
        
        console.print(Panel(stats, title="[bold yellow]Cache Statistics"))
        
        # 최근 활동 테이블
        recent_entries = collection.query(
            expr="",
            output_fields=["created_at", "email", "project_id", "query_text"],
            limit=5,
            sort_fields=["created_at"],
            sort_orders=["DESC"]  # 최신순 정렬
        )
        
        if recent_entries:
            recent_table = Table(
                title="Recent Entries",
                border_style="bright_blue",
                show_header=True
            )
            recent_table.add_column("Created At", style="cyan")
            recent_table.add_column("Email", style="white")
            recent_table.add_column("Project", style="green")
            recent_table.add_column("Query", style="yellow", width=40)
            
            for entry in recent_entries:
                recent_table.add_row(
                    entry["created_at"],
                    entry["email"],
                    entry["project_id"],
                    entry.get("query_text", "")[:40] + "..." if entry.get("query_text") else ""
                )
            
            console.print(Panel(recent_table, title="[bold yellow]Recent Activity"))
            
        # 프로젝트별 통계 테이블
        if project_count:
            project_table = Table(
                title="Project Statistics",
                border_style="bright_blue",
                show_header=True
            )
            project_table.add_column("Project ID", style="cyan")
            project_table.add_column("Cache Count", style="white", justify="right")
            
            for project_id, count in sorted(
                project_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:  # Top 5 projects
                project_table.add_row(project_id, str(count))
            
            console.print(Panel(project_table, title="[bold yellow]Top Projects"))
            
    except Exception as e:
        logger.error(f"Failed to display statistics: {e}")
        console.print(Panel(
            "[red]Error getting statistics: {e}[/red]",
            title="Error",
            border_style="red"
        ))

async def drop_collection(cache_manager: CacheManager) -> None:
    """컬렉션 완전 삭제"""
    console.clear()
    console.print(create_header())
    
    # 경고 메시지 표시
    warning_panel = Panel(
        Group(
            Text("⚠️  [bold red]DANGER ZONE[/bold red] ⚠️", justify="center"),
            Text(""),
            Text("[bold red]This action will:[/bold red]", justify="center"),
            Text("• Permanently delete the entire collection"),
            Text("• Remove all cached data"),
            Text("• This action CANNOT be undone"),
            Text(""),
            Text("[bold yellow]Are you absolutely sure?[/bold yellow]", justify="center")
        ),
        title="[bold red]Drop Collection",
        border_style="red"
    )
    console.print(warning_panel)
    
    # 확인을 위해 컬렉션 이름 입력 요구
    console.print("\n[yellow]To confirm, type the collection name:[/yellow]")
    collection_name = Prompt.ask("Collection name")
    
    if collection_name == cache_manager.collection_name:
        if Confirm.ask("\nAre you absolutely sure you want to drop the collection?"):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Dropping collection...", total=None)
                try:
                    if utility.has_collection(collection_name):
                        utility.drop_collection(collection_name)
                        console.print(Panel(
                            "[bold green]✅ Collection has been permanently deleted![/bold green]",
                            border_style="green"
                        ))
                    else:
                        console.print(Panel(
                            "[bold yellow]Collection does not exist![/bold yellow]",
                            border_style="yellow"
                        ))
                except Exception as e:
                    console.print(Panel(
                        f"[bold red]Failed to drop collection: {str(e)}[/bold red]",
                        border_style="red"
                    ))
    else:
        console.print(Panel(
            "[bold red]Collection name does not match. Operation cancelled.[/bold red]",
            border_style="red"
        ))
    
    input("\nPress Enter to continue...")

async def interactive_menu():
    cache_manager = CacheManager()
    page = 1
    
    while True:
        console.clear()
        console.print(create_header())
        console.print(create_menu())
        console.print(create_footer())
        
        choice = Prompt.ask(
            "\nSelect an option (or 'q' to quit)",
            choices=["1", "2", "3", "4", "5", "6", "7", "q"],
            show_choices=False
        ).lower()
        
        if choice == 'q':
            break
            
        if choice == "1":
            while True:
                entries, total_count = await cache_manager.get_cache_entries(page)
                total_pages = math.ceil(total_count / cache_manager.page_size)
                
                console.clear()
                console.print(create_header())
                console.print(display_entries(entries, page, total_pages))
                console.print(create_footer())
                
                navigation = Table.grid(padding=1)
                nav_items = []
                if page > 1:
                    nav_items.append("p. previous")
                if page < total_pages:
                    nav_items.append("n. next")
                nav_items.append("number. view detail")
                nav_items.append("b. back")
                nav_items.append("q. quit")
                
                navigation.add_row(*nav_items)
                console.print(Panel(navigation, border_style="blue"))
                
                nav_choice = Prompt.ask(
                    "Select option or entry number",
                    show_choices=False
                ).lower()
                
                if nav_choice == "q":
                    return
                elif nav_choice == "p" and page > 1:
                    page -= 1
                elif nav_choice == "n" and page < total_pages:
                    page += 1
                elif nav_choice == "b":
                    break
                elif nav_choice.isdigit():
                    entry_idx = int(nav_choice)
                    if 1 <= entry_idx <= len(entries):
                        while True:
                            console.clear()
                            console.print(create_header())
                            console.print(display_entry_detail(entries[entry_idx - 1]))
                            console.print(create_footer())
                            
                            detail_choice = Prompt.ask(
                                "Select action",
                                choices=["d", "b"],
                                show_choices=False
                            ).lower()
                            
                            if detail_choice == "d":
                                console.print("\n[bold red]⚠️  Warning: This action cannot be undone![/bold red]")
                                if Confirm.ask(
                                    f"\nAre you sure you want to delete entry {entries[entry_idx - 1]['id']}?"
                                ):
                                    with Progress(
                                        SpinnerColumn(),
                                        TextColumn("[progress.description]{task.description}"),
                                        transient=True,
                                    ) as progress:
                                        progress.add_task(description="Deleting entry...", total=None)
                                        success = await cache_manager.delete_entry(entries[entry_idx - 1]['id'])
                                    
                                    if success:
                                        console.print(Panel(
                                            "[green]Entry deleted successfully[/green]",
                                            border_style="green"
                                        ))
                                        time.sleep(1)
                                        break  # 상세 보기 종료
                                    else:
                                        console.print(Panel(
                                            "[red]Failed to delete entry[/red]",
                                            border_style="red"
                                        ))
                                        time.sleep(1)
                            elif detail_choice == "b":
                                break  # 상세 보기 종료
                    else:
                        console.print("[red]Invalid entry number[/red]")
                        time.sleep(1)

        elif choice == "2":
            search_query = Prompt.ask("\nEnter search term")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Searching...", total=None)
                results = await cache_manager.search_entries(search_query)
            
            if results:
                while True:  # 검색 결과 처리를 위한 루프
                    console.clear()
                    console.print(create_header())
                    console.print(display_entries(results, 1, 1))
                    
                    # 검색 결과 네비게이션 옵션
                    navigation = Table.grid(padding=1)
                    nav_items = [
                        "number. view detail",
                        "b. back to menu",
                        "q. quit"
                    ]
                    navigation.add_row(*nav_items)
                    console.print(Panel(navigation, border_style="blue"))
                    
                    nav_choice = Prompt.ask(
                        "Select option or entry number",
                        show_choices=False
                    ).lower()
                    
                    if nav_choice == "q":
                        return
                    elif nav_choice == "b":
                        break
                    elif nav_choice.isdigit():
                        entry_idx = int(nav_choice)
                        if 1 <= entry_idx <= len(results):
                            while True:  # 상세 보기 루프
                                console.clear()
                                console.print(create_header())
                                console.print(display_entry_detail(results[entry_idx - 1]))
                                console.print(create_footer())
                                
                                detail_choice = Prompt.ask(
                                    "Select action",
                                    choices=["d", "b"],
                                    show_choices=False
                                ).lower()
                                
                                if detail_choice == "d":
                                    if Confirm.ask(
                                        f"\nAre you sure you want to delete entry {results[entry_idx - 1]['id']}?"
                                    ):
                                        with Progress(
                                            SpinnerColumn(),
                                            TextColumn("[progress.description]{task.description}"),
                                            transient=True,
                                        ) as progress:
                                            progress.add_task(description="Deleting entry...", total=None)
                                            success = await cache_manager.delete_entry(results[entry_idx - 1]['id'])
                                        
                                        if success:
                                            console.print(Panel(
                                                "[green]Entry deleted successfully[/green]",
                                                border_style="green"
                                            ))
                                            # 결과 목록에서도 삭제
                                            results.pop(entry_idx - 1)
                                            time.sleep(1)
                                            break  # 상세 보기 종료
                                        else:
                                            console.print(Panel(
                                                "[red]Failed to delete entry[/red]",
                                                border_style="red"
                                            ))
                                            time.sleep(1)
                                elif detail_choice == "b":
                                    break  # 상세 보기 종료
                        else:
                            console.print("[red]Invalid entry number[/red]")
                            time.sleep(1)
            else:
                console.print(Panel(
                    "[yellow]No matching entries found[/yellow]",
                    border_style="yellow"
                ))
                input("\nPress Enter to continue...")

        elif choice == "3":
            entry_id = Prompt.ask("\nEnter entry ID to delete")
            if Confirm.ask(f"Are you sure you want to delete entry {entry_id}?"):
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    progress.add_task(description="Deleting entry...", total=None)
                    success = await cache_manager.delete_entry(entry_id)
                
                if success:
                    console.print(Panel(
                        "[green]Entry deleted successfully[/green]",
                        border_style="green"
                    ))
                else:
                    console.print(Panel(
                        "[red]Failed to delete entry[/red]",
                        border_style="red"
                    ))
                input("\nPress Enter to continue...")

        elif choice == "4":
            console.clear()
            console.print(create_header())
            await display_statistics(cache_manager)
            input("\nPress Enter to continue...")

        elif choice == "5":
            console.clear()
            console.print(create_header())
            console.print(Panel(
                "[bold red]⚠️  Warning: This will delete all cached data![/bold red]\n\n"
                "This action will:\n"
                "• Delete all existing cache entries\n"
                "• Reset the collection schema\n"
                "• Fix dimension mismatch issues\n",
                title="Reset Collection",
                border_style="red"
            ))
            
            if Confirm.ask("Are you sure you want to reset the entire cache collection?", default=False):
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    progress.add_task(description="Resetting collection...", total=None)
                    await cache_manager.collection.drop()
                
                console.print(Panel(
                    "[bold green]✅ Cache collection has been successfully reset![/bold green]",
                    border_style="green"
                ))
                input("\nPress Enter to continue...")

        elif choice == "6":
            await drop_collection(cache_manager)

        elif choice == "7":
            break

    console.clear()
    console.print(create_header())
    console.print(Panel(
        "[bold green]Thank you for using LogosAI Cache Management System![/bold green]",
        border_style="green"
    ))
    console.print(create_footer())

@click.command()
def main():
    """LogosAI Cache Management System"""
    try:
        asyncio.run(interactive_menu())
    except KeyboardInterrupt:
        console.print("\n[bold green]Goodbye![/bold green]")
    except Exception as e:
        console.print(Panel(
            f"[red]An error occurred: {e}[/red]",
            title="Error",
            border_style="red"
        ))
    finally:
        connections.disconnect("default")

if __name__ == "__main__":
    main() 