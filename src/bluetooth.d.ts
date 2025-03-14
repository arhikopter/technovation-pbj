// Type definitions for Web Bluetooth API
interface BluetoothRequestDeviceOptions {
  filters?: Array<{
    services?: string[];
    name?: string;
    namePrefix?: string;
  }>;
  optionalServices?: string[];
  acceptAllDevices?: boolean;
}

// Needed to make TypeScript aware of the Web Bluetooth API without conflicts
interface BluetoothDevice {
  id: string;
  name?: string;
  gatt?: {
    connected: boolean;
    connect(): Promise<any>;
  };
} 