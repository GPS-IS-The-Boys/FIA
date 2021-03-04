package model;

import java.util.Map;
import java.util.Objects;

/**
 * Record o Istanza che rappresenta il singolo elemento del dataset
 */
public class Record {
  private final String descrizione;
  private final Map<String, Double> features;

  /**
   * Costruttore del record
   * @param descrizione
   * @param features
   */
  public Record(String descrizione, Map<String, Double> features) {
    this.descrizione = descrizione;
    this.features = features;
  }

  /**
   * Torna la descrizione del record
   * @return
   */
  public String getDescrizione() {
    return descrizione;
  }

  /**
   * Torna la feature del record
   * @return
   */
  public Map<String, Double> getFeatures() {
    return features;
  }

  @Override
  public String toString() {
    return "Record{" +
        "descrizione='" + descrizione + '\'' +
        ", features=" + features +
        '}';
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof Record)) return false;
    Record record = (Record) o;
    return Objects.equals(getDescrizione(), record.getDescrizione()) &&
        Objects.equals(getFeatures(), record.getFeatures());
  }

  @Override
  public int hashCode() {
    return Objects.hash(getDescrizione(), getFeatures());
  }
}
