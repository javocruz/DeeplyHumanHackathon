import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from supabase import create_client, Client
import os
import warnings
warnings.filterwarnings('ignore')

# ========================= 
# CONFIG SUPABASE
# ========================= 
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ziafykczhnjcfcmxleba.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InppYWZ5a2N6aG5qY2ZjbXhsZWJhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTAwNjAwNywiZXhwIjoyMDg0NTgyMDA3fQ.pdoV-n6DRJFk9WVHRGSBntBWy-CVRSRCJHGMnhLnscs")
METADATA_PATH = "column_metadata.csv"

# Configuración
MAX_COLS = 5
TOP_N_FOR_RANDOM = 20
MIN_COMPLETENESS = 0.6
PREFERRED_START_YEAR = 2000
MAX_YEAR = 2025

class SupabaseRecommender:
    def __init__(self):
        """Inicializa el sistema de recomendación con Supabase"""
        print("🔌 Conectando a Supabase...")
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        print("📋 Cargando metadata...")
        self.metadata = pd.read_csv(METADATA_PATH)
        
        print("✅ Sistema inicializado\n")
    
    def score_columns(self, user_scores: Dict[str, float]) -> pd.DataFrame:
        """Puntúa columnas basado en prioridades del usuario"""
        df = self.metadata.copy()
        df["dynamic_score"] = 0.0
        
        # Normalizar scores del usuario
        max_score = max(user_scores.values())
        user_norm = {k.lower(): v/max_score for k, v in user_scores.items()}
        
        # Calcular score para cada columna
        for idx, row in df.iterrows():
            score = 0.0
            
            # Score por label primario
            primary = str(row["primary_label"]).lower()
            if primary in user_norm:
                score += row["confidence"] * user_norm[primary]
            
            # Score por labels secundarios
            if isinstance(row["secondary_labels"], str):
                try:
                    secondary_list = eval(row["secondary_labels"])
                    for sec in secondary_list:
                        sec = sec.lower()
                        if sec in user_norm:
                            score += 0.5 * row["confidence"] * user_norm[sec]
                except:
                    pass
            
            df.at[idx, "dynamic_score"] = score
        
        return df.sort_values("dynamic_score", ascending=False)
    
    def select_columns(self, scored_df: pd.DataFrame, max_cols: int = MAX_COLS) -> List[str]:
        """Selecciona columnas con aleatoriedad ponderada"""
        df_top = scored_df.head(TOP_N_FOR_RANDOM).copy()
        
        scores = df_top["dynamic_score"].values
        probs = scores / scores.sum() if scores.sum() > 0 else np.ones(len(scores))/len(scores)
        
        n_select = min(max_cols, len(df_top))
        selected_indices = np.random.choice(len(df_top), size=n_select, replace=False, p=probs)
        
        return df_top.iloc[selected_indices]["column"].tolist()
    
    def fetch_country_data(self, country_code: str, columns: List[str]) -> Optional[pd.DataFrame]:
        """
        Obtiene datos de Supabase para un país y columnas específicas
        """
        try:
            # Construir select con columnas necesarias
            select_cols = ['geo', 'time'] + columns
            select_str = ','.join(select_cols)
            
            print(f"🔍 Consultando Supabase para '{country_code}'...")
            
            # Query optimizada con filtros
            response = (
                self.supabase.table('country_data')
                .select(select_str)
                .eq('geo', country_code.lower())
                .gte('time', PREFERRED_START_YEAR)
                .lte('time', MAX_YEAR)
                .order('time', desc=False)
                .execute()
            )
            
            if not response.data or len(response.data) == 0:
                print(f"   ⚠️  No hay datos para '{country_code}' en rango {PREFERRED_START_YEAR}-{MAX_YEAR}")
                
                # Intentar sin filtro de año
                print(f"   🔄 Intentando con rango ampliado...")
                response = (
                    self.supabase.table('country_data')
                    .select(select_str)
                    .eq('geo', country_code.lower())
                    .order('time', desc=False)
                    .execute()
                )
                
                if not response.data:
                    print(f"   ❌ No hay datos para '{country_code}'")
                    return None
            
            df = pd.DataFrame(response.data)
            print(f"   ✅ {len(df)} registros obtenidos")
            return df
            
        except Exception as e:
            print(f"   ❌ Error al consultar Supabase: {e}")
            return None
    
    def analyze_column_quality(self, df: pd.DataFrame, column: str) -> Optional[Dict]:
        """Analiza la calidad de datos de una columna"""
        if column not in df.columns:
            return None
        
        # Filtrar por rango preferido
        df_filtered = df[
            (df['time'] >= PREFERRED_START_YEAR) & 
            (df['time'] <= MAX_YEAR)
        ].copy()
        
        # Si no hay datos en rango preferido, usar todos
        if len(df_filtered) == 0:
            df_filtered = df.copy()
        
        if len(df_filtered) == 0:
            return None
        
        # Análisis de completitud
        non_null = df_filtered[column].notna()
        completeness = non_null.sum() / len(df_filtered) if len(df_filtered) > 0 else 0
        
        # Si muy baja completitud, buscar mejor rango continuo
        if completeness < MIN_COMPLETENESS:
            df_non_null = df_filtered[non_null].copy()
            if len(df_non_null) < 5:  # Mínimo 5 puntos
                return None
            
            start_year = int(df_non_null['time'].min())
            end_year = int(df_non_null['time'].max())
            data = df_non_null[['time', column]].copy()
        else:
            # Usar rango completo
            start_year = int(df_filtered['time'].min())
            end_year = int(df_filtered['time'].max())
            data = df_filtered[non_null][['time', column]].copy()
        
        return {
            'column': column,
            'start_year': start_year,
            'end_year': end_year,
            'completeness': float(completeness),
            'data_points': len(data),
            'data': data
        }
    
    def get_recommendations(self, country_code: str, user_scores: Dict[str, float]) -> Optional[Dict]:
        """
        Pipeline completo de recomendación
        
        Args:
            country_code: Código del país (ej: 'esp', 'usa')
            user_scores: Dict con prioridades del usuario
            
        Returns:
            Dict con recomendaciones y datos
        """
        print("=" * 80)
        print(f"🚀 ANÁLISIS PARA: {country_code.upper()}")
        print("=" * 80)
        
        # 1. Scoring y selección de columnas (en memoria - rápido)
        print("\n1️⃣ Puntuando columnas...")
        scored_df = self.score_columns(user_scores)
        selected_columns = self.select_columns(scored_df)
        print(f"   ✅ Seleccionadas: {selected_columns}")
        
        # 2. Obtener datos de Supabase
        print("\n2️⃣ Obteniendo datos de Supabase...")
        df_country = self.fetch_country_data(country_code, selected_columns)
        
        if df_country is None:
            print("\n❌ No se pudieron obtener datos")
            return None
        
        # 3. Analizar calidad de cada columna
        print("\n3️⃣ Analizando calidad de datos...")
        results = []
        
        for col in selected_columns:
            quality = self.analyze_column_quality(df_country, col)
            
            if quality is None:
                print(f"   ⚠️  '{col}' - Datos insuficientes")
                continue
            
            results.append(quality)
            print(f"   ✅ '{col}' - {quality['start_year']}-{quality['end_year']} ({quality['completeness']*100:.1f}%)")
        
        if not results:
            print("\n❌ Ninguna columna tiene datos suficientes")
            return None
        
        # 4. Generar resumen
        summary = {
            'total_columns': len(results),
            'avg_completeness': float(np.mean([r['completeness'] for r in results])),
            'date_range': {
                'start': int(min([r['start_year'] for r in results])),
                'end': int(max([r['end_year'] for r in results]))
            },
            'total_data_points': int(sum([r['data_points'] for r in results]))
        }
        
        print(f"\n✅ {len(results)} columnas con datos válidos")
        
        return {
            'country': country_code,
            'columns': selected_columns,
            'data': results,
            'summary': summary
        }
    
    def print_results(self, result: Dict):
        """Imprime resultados de forma legible"""
        if not result:
            return
        
        print("\n" + "=" * 80)
        print("📊 RESUMEN EJECUTIVO")
        print("=" * 80)
        print(f"🌍 País: {result['country'].upper()}")
        print(f"📈 Columnas con datos: {result['summary']['total_columns']}")
        print(f"✓  Completitud promedio: {result['summary']['avg_completeness']*100:.1f}%")
        print(f"📅 Periodo: {result['summary']['date_range']['start']}-{result['summary']['date_range']['end']}")
        print(f"📊 Total de datos: {result['summary']['total_data_points']} registros")
        
        print("\n" + "=" * 80)
        print("📈 DATOS DETALLADOS")
        print("=" * 80)
        
        for item in result['data']:
            print(f"\n🔹 {item['column']}")
            print(f"   ⏱️  Periodo: {item['start_year']}-{item['end_year']}")
            print(f"   ✓  Completitud: {item['completeness']*100:.1f}%")
            print(f"   📊 Registros: {item['data_points']}")
            
            # Mostrar últimos 5 valores
            print(f"\n   Últimos 5 valores:")
            tail_data = item['data'].tail()
            for _, row in tail_data.iterrows():
                print(f"      {int(row['time'])}: {row[item['column']]}")
    
    def export_json(self, result: Dict, filename: str = None):
        """Exporta resultados a JSON"""
        import json
        
        export_data = {
            'country': result['country'],
            'summary': result['summary'],
            'columns': result['columns'],
            'data': []
        }
        
        for item in result['data']:
            export_data['data'].append({
                'column': item['column'],
                'start_year': item['start_year'],
                'end_year': item['end_year'],
                'completeness': item['completeness'],
                'data_points': item['data_points'],
                'records': item['data'].to_dict('records')
            })
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Exportado a: {filename}")
        
        return export_data


# ========================= 
# EJEMPLO DE USO
# ========================= 
if __name__ == "__main__":
    # Inicializar sistema
    recommender = SupabaseRecommender()
    
    # Definir prioridades del usuario
    user_priorities = {
        "cultural": 10,
        "economic": 10,
        "environmental": 10,
        "mental health": 10,
        "physical health": 8.8,
        "social wellbeing": 10
    }
    
    # Obtener recomendaciones
    result = recommender.get_recommendations("esp", user_priorities)
    
    # Mostrar resultados
    if result:
        recommender.print_results(result)
        
        # Exportar a JSON
        recommender.export_json(result, f"{result['country']}_recommendations.json")
        
        print("\n" + "=" * 80)
        print("✅ PROCESO COMPLETADO")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ NO SE OBTUVIERON RESULTADOS")
        print("=" * 80)
        print("\n💡 Sugerencias:")
        print("   1. Verifica que el país existe: python diagnose_supabase_v2.py")
        print("   2. Verifica las credenciales de Supabase")
        print("   3. Verifica que la tabla 'country_data' tiene datos")
