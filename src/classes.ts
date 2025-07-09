export interface ObjectDetectionClass {
  name: string;
  id: number;
  displayName: string;
}

export const CLASSES: { [key: number]: ObjectDetectionClass } = {
  0: { name: "light_vehicle", id: 0, displayName: "LightVehicle" },
  1: { name: "person", id: 1, displayName: "Person" },
  2: { name: "building", id: 2, displayName: "Building" },
  3: { name: "upole", id: 3, displayName: "UPole" },
  4: { name: "boat", id: 4, displayName: "Boat" },
  5: { name: "bike", id: 5, displayName: "Bike" },
  6: { name: "container", id: 6, displayName: "Container" },
  7: { name: "truck", id: 7, displayName: "Truck" },
  8: { name: "gastank", id: 8, displayName: "Gastank" },
  // 9번은 없음 (모델 설명에서 제외)
  10: { name: "digger", id: 10, displayName: "Digger" },
  11: { name: "solarpanels", id: 11, displayName: "Solarpanels" },
  12: { name: "bus", id: 12, displayName: "Bus" },
};
